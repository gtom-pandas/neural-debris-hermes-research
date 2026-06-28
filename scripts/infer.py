"""
Apply Neural Debris confidence repair checkpoints to a detection CSV.

The input CSV should contain one row per candidate detection with:
  image_id, confidence, x_min, y_min, x_max, y_max

An NPZ file supplies aligned detection features. The script averages one or more
repair heads, blends them conservatively with the original confidence, filters
low-confidence boxes and writes a Kaggle-style PredictionString submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class ConfidenceRepairHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_models(paths: list[Path], device: torch.device) -> list[nn.Module]:
    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device)
        model = ConfidenceRepairHead(checkpoint["input_dim"], checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def format_prediction_string(group: pd.DataFrame) -> str:
    parts: list[str] = []
    for row in group.itertuples(index=False):
        parts.extend(
            [
                f"{row.confidence:.6f}",
                f"{row.x_min:.2f}",
                f"{row.y_min:.2f}",
                f"{row.x_max:.2f}",
                f"{row.y_max:.2f}",
            ]
        )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="Candidate detection CSV.")
    parser.add_argument("--features", required=True, help="Aligned NPZ with features array.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Repair checkpoint. Can be repeated.")
    parser.add_argument("--blend-weight", type=float, default=0.25, help="Weight of repair confidence vs original confidence.")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    detections = pd.read_csv(args.detections)
    required = {"image_id", "confidence", "x_min", "y_min", "x_max", "y_max"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Missing columns in detection CSV: {sorted(missing)}")

    features = np.load(args.features)["features"].astype("float32")
    if len(features) != len(detections):
        raise ValueError(f"Feature rows ({len(features)}) do not match detections ({len(detections)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models([Path(p) for p in args.checkpoint], device)
    with torch.no_grad():
        tensor_x = torch.from_numpy(features).to(device)
        repaired = [torch.sigmoid(model(tensor_x)).detach().cpu().numpy() for model in models]
    repair_conf = np.mean(repaired, axis=0)

    original = detections["confidence"].to_numpy(dtype="float32")
    detections["confidence"] = (1.0 - args.blend_weight) * original + args.blend_weight * repair_conf
    detections = detections[detections["confidence"] >= args.threshold].copy()
    detections = (
        detections.sort_values(["image_id", "confidence"], ascending=[True, False])
        .groupby("image_id", as_index=False)
        .head(args.top_k)
    )

    image_ids = pd.DataFrame({"image_id": pd.read_csv(args.detections)["image_id"].drop_duplicates()})
    pred = detections.groupby("image_id").apply(format_prediction_string).rename("PredictionString").reset_index()
    submission = image_ids.merge(pred, on="image_id", how="left").fillna({"PredictionString": ""})
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(submission)} images")


if __name__ == "__main__":
    main()
