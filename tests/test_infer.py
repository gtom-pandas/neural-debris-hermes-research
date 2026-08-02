from __future__ import annotations

import pandas as pd
import pytest

from infer import build_submission, format_prediction_string


def test_prediction_string_uses_xywh_contract() -> None:
    detections = pd.DataFrame(
        [{"image_id": "a", "confidence": 0.75, "x": 10, "y": 20, "width": 30, "height": 40}]
    )
    assert format_prediction_string(detections) == "0.750000 10.00 20.00 30.00 40.00"


def test_submission_preserves_sample_order_and_empty_images() -> None:
    detections = pd.DataFrame(
        [{"image_id": "b", "confidence": 0.6, "x": 1, "y": 2, "width": 3, "height": 4}]
    )
    sample = pd.DataFrame(
        {"image_id": ["a", "b", "c"], "PredictionString": ["template", "template", "template"]}
    )

    result = build_submission(detections, sample)

    assert result["image_id"].tolist() == ["a", "b", "c"]
    assert result["PredictionString"].tolist() == ["", "0.600000 1.00 2.00 3.00 4.00", ""]


def test_submission_rejects_duplicate_image_ids() -> None:
    sample = pd.DataFrame({"image_id": ["a", "a"], "PredictionString": ["", ""]})
    with pytest.raises(ValueError, match="must be unique"):
        build_submission(pd.DataFrame(columns=["image_id"]), sample)
