"""
Controlled unlearning reference trainer for Neural Debris.

This script documents the model-repair direction used in the research repo
without committing Kaggle datasets or private checkpoints. It trains a compact
confidence correction head on exported detection-level features.

Expected NPZ arrays:
  features: float32 [n_detections, n_features]
  poison:   float32 [n_detections] where 1 marks detections to suppress
  keep:     float32 [n_detections] where 1 marks detections to preserve
  teacher:  float32 [n_detections] original detector confidence

The loss combines poison suppression and preserve distillation. It is a small,
auditable proxy for the larger RetinaNet/Detectron2 repair notebooks stored
under kaggle_kernels/neural_debris/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    features: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    poison_weight: float
    preserve_weight: float
    folds: int
    seed: int


class ConfidenceRepairHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    features = payload["features"].astype("float32")
    poison = payload["poison"].astype("float32")
    keep = payload["keep"].astype("float32")
    teacher = payload["teacher"].astype("float32")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    return features, poison, keep, teacher


def repair_loss(
    logits: torch.Tensor,
    poison: torch.Tensor,
    keep: torch.Tensor,
    teacher: torch.Tensor,
    poison_weight: float,
    preserve_weight: float,
) -> torch.Tensor:
    target = torch.clamp(teacher, 0.0, 1.0)
    probs = torch.sigmoid(logits)

    suppress = (probs * poison).mean()
    preserve = ((probs - target).pow(2) * keep).sum() / torch.clamp(keep.sum(), min=1.0)
    calibration = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return calibration + poison_weight * suppress + preserve_weight * preserve


def run_fold(
    features: np.ndarray,
    poison: np.ndarray,
    keep: np.ndarray,
    teacher: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    cfg: TrainConfig,
    fold: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConfidenceRepairHead(features.shape[1], cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.from_numpy(features[train_idx]),
        torch.from_numpy(poison[train_idx]),
        torch.from_numpy(keep[train_idx]),
        torch.from_numpy(teacher[train_idx]),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    valid_x = torch.from_numpy(features[valid_idx]).to(device)
    valid_poison = torch.from_numpy(poison[valid_idx]).to(device)
    valid_keep = torch.from_numpy(keep[valid_idx]).to(device)
    valid_teacher = torch.from_numpy(teacher[valid_idx]).to(device)

    best_loss = float("inf")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_poison, batch_keep, batch_teacher in loader:
            batch_x = batch_x.to(device)
            batch_poison = batch_poison.to(device)
            batch_keep = batch_keep.to(device)
            batch_teacher = batch_teacher.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = repair_loss(
                logits,
                batch_poison,
                batch_keep,
                batch_teacher,
                cfg.poison_weight,
                cfg.preserve_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x)
            valid_loss = repair_loss(
                valid_logits,
                valid_poison,
                valid_keep,
                valid_teacher,
                cfg.poison_weight,
                cfg.preserve_weight,
            )
            valid_probs = torch.sigmoid(valid_logits)
            poison_mean = float(valid_probs[valid_poison > 0.5].mean().detach().cpu()) if (valid_poison > 0.5).any() else 0.0
            keep_delta = float((valid_probs - valid_teacher).abs()[valid_keep > 0.5].mean().detach().cpu()) if (valid_keep > 0.5).any() else 0.0

        valid_loss_value = float(valid_loss.detach().cpu())
        if valid_loss_value < best_loss:
            best_loss = valid_loss_value
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": features.shape[1],
                    "hidden_dim": cfg.hidden_dim,
                    "fold": fold,
                    "config": asdict(cfg),
                },
                output_dir / f"confidence_repair_fold{fold}.pt",
            )

        print(
            json.dumps(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "valid_loss": valid_loss_value,
                    "valid_poison_confidence": poison_mean,
                    "valid_keep_abs_delta": keep_delta,
                }
            )
        )

    return {"fold": fold, "best_valid_loss": best_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="artifacts/confidence_repair")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--poison-weight", type=float, default=0.05)
    parser.add_argument("--preserve-weight", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)
    features, poison, keep, teacher = load_features(Path(cfg.features))

    stratify = np.clip(poison.astype(int) + keep.astype(int), 0, 2)
    splitter = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    metrics = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features, stratify), start=1):
        metrics.append(run_fold(features, poison, keep, teacher, train_idx, valid_idx, cfg, fold))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
