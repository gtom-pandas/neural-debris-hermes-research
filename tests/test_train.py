from __future__ import annotations

import numpy as np
import pytest
import torch

from train import load_features, repair_loss, stratified_folds


def test_load_features_validates_aligned_contract(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        features=np.ones((4, 3), dtype=np.float32),
        poison=np.array([1, 0, 0, 0], dtype=np.float32),
        keep=np.array([0, 1, 1, 1], dtype=np.float32),
        teacher=np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
    )

    features, poison, keep, teacher = load_features(path)

    assert features.shape == (4, 3)
    assert poison.shape == keep.shape == teacher.shape == (4,)


def test_load_features_rejects_invalid_teacher_confidence(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        features=np.ones((2, 2), dtype=np.float32),
        poison=np.zeros(2, dtype=np.float32),
        keep=np.ones(2, dtype=np.float32),
        teacher=np.array([0.5, 1.2], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_features(path)


def test_repair_loss_penalizes_poison_confidence() -> None:
    poison = torch.tensor([1.0, 0.0])
    keep = torch.tensor([0.0, 1.0])
    teacher = torch.tensor([0.9, 0.8])
    low_poison = repair_loss(torch.tensor([-3.0, 1.4]), poison, keep, teacher, 2.0, 1.0)
    high_poison = repair_loss(torch.tensor([3.0, 1.4]), poison, keep, teacher, 2.0, 1.0)
    assert low_poison < high_poison


def test_stratified_folds_are_complete_and_deterministic() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    first = list(stratified_folds(labels, n_splits=2, seed=42))
    second = list(stratified_folds(labels, n_splits=2, seed=42))
    assert all(np.array_equal(a, b) for pair_a, pair_b in zip(first, second) for a, b in zip(pair_a, pair_b))
    assert sorted(np.concatenate([valid for _, valid in first]).tolist()) == list(range(len(labels)))
