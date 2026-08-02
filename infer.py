"""
Apply Neural Debris confidence repair checkpoints to a detection CSV.

The input CSV should contain one row per candidate detection with:
  image_id, confidence, x, y, width, height

An NPZ file supplies aligned detection features. The script averages one or more
repair heads, blends them conservatively with the original confidence, filters
low-confidence boxes and writes a Kaggle-style PredictionString submission.
The sample submission is the source of truth for image order and empty rows.
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
    expected_input_dim: int | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        input_dim = int(checkpoint["input_dim"])
        if expected_input_dim is None:
            expected_input_dim = input_dim
        elif input_dim != expected_input_dim:
            raise ValueError("All checkpoints must use the same input dimension")
        model = ConfidenceRepairHead(input_dim, int(checkpoint["hidden_dim"]))
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
                f"{row.x:.2f}",
                f"{row.y:.2f}",
                f"{row.width:.2f}",
                f"{row.height:.2f}",
            ]
        )
    return " ".join(parts)


def build_submission(detections: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    required_sample = {"image_id", "PredictionString"}
    missing_sample = required_sample.difference(sample.columns)
    if missing_sample:
        raise ValueError(f"Missing columns in sample submission: {sorted(missing_sample)}")
    if sample["image_id"].duplicated().any():
        raise ValueError("sample submission image_id values must be unique")

    predictions = {
        image_id: format_prediction_string(group)
        for image_id, group in detections.groupby("image_id", sort=False)
    }
    submission = sample.copy()
    submission["PredictionString"] = submission["image_id"].map(predictions).fillna("")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="Candidate detection CSV.")
    parser.add_argument("--features", required=True, help="Aligned NPZ with features array.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Repair checkpoint. Can be repeated.")
    parser.add_argument("--sample-submission", required=True, help="Official sample submission CSV.")
    parser.add_argument("--blend-weight", type=float, default=0.25, help="Weight of repair confidence vs original confidence.")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    detections = pd.read_csv(args.detections)
    required = {"image_id", "confidence", "x", "y", "width", "height"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Missing columns in detection CSV: {sorted(missing)}")

    features = np.load(args.features, allow_pickle=False)["features"].astype("float32")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite 2D array")
    if len(features) != len(detections):
        raise ValueError(f"Feature rows ({len(features)}) do not match detections ({len(detections)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models([Path(p) for p in args.checkpoint], device)
    if models and features.shape[1] != models[0].net[0].normalized_shape[0]:
        raise ValueError("Feature dimension does not match checkpoint input dimension")
    with torch.no_grad():
        tensor_x = torch.from_numpy(features).to(device)
        repaired = [torch.sigmoid(model(tensor_x)).detach().cpu().numpy() for model in models]
    repair_conf = np.mean(repaired, axis=0)

    original = detections["confidence"].to_numpy(dtype="float32")
    detections["confidence"] = (1.0 - args.blend_weight) * original + args.blend_weight * repair_conf
    if not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("blend-weight must be between 0 and 1")
    if args.top_k < 1:
        raise ValueError("top-k must be at least 1")
    if (detections[["width", "height"]] <= 0).any().any():
        raise ValueError("width and height must be positive")
    detections = detections[detections["confidence"] >= args.threshold].copy()
    detections = (
        detections.sort_values(["image_id", "confidence"], ascending=[True, False])
        .groupby("image_id", as_index=False)
        .head(args.top_k)
    )

    sample = pd.read_csv(args.sample_submission, keep_default_na=False)
    unknown_ids = set(detections["image_id"]).difference(sample["image_id"])
    if unknown_ids:
        raise ValueError(f"Detections contain image IDs absent from sample submission: {len(unknown_ids)}")
    submission = build_submission(detections, sample)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(submission)} images")


if __name__ == "__main__":
    main()
