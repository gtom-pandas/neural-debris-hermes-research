# Targeted Model Repair - Neural Debris

This repository accompanies my study of the Kaggle competition
[Neural Debris Removal in Streak Detection Models](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models).
It documents the repair of a poisoned RetinaNet detector when the clean model
and test labels are hidden.

**Best public maCADD:** 219.4257<br>
**Best private maCADD:** 306.3502<br>
**Final rank:** 89 / 567 teams

Lower maCADD is better. The public-best candidate scored 308.0971 privately;
the best private result came from a pruning plus short EWC fine-tune.

## Research trajectory

```text
aggressive suppression (449.95)
          |
short empty-label fine-tuning (259.79)
          |
preservation losses and interpolation (257.12)
          |
pruning / EWC reproductions (245.30)
          |
Sanidhya Phase B (229.19 public, 306.35 private)
          |
metric-aware confidence selection (219.43 public, 308.10 private)
```

The central lesson is that suppression alone causes collateral damage. Reliable
repair needs a frozen teacher, dense preservation of non-poison anchors and box
deltas, a poison mask aligned with RetinaNet anchors, and synthetic validation
that measures both forgetting and clean-streak retention.

## Files

- `paper/neural_debris_technical_report.pdf` is the 13-page ORCID-linked research paper. It includes exact EDA, the reconstructed ResNet-50-FPN RetinaNet architecture, the complete pruning/EWC/inference stack, hyperparameter failures, output-level ablations and leaderboard forensics.
- `train.py` trains the released detection-level confidence repair head.
- `infer.py` ensembles repair checkpoints and writes a contract-checked Kaggle submission.
- `SOURCES.md` records the competition, research references and external solutions.
- `CITATION.cff` provides citation metadata.
- `tests/` contains focused contract and regression tests.

## Training

`train.py` expects an NPZ file with aligned detection-level arrays:

- `features`: float matrix `[n_detections, n_features]`;
- `poison`: binary suppression mask;
- `keep`: binary preservation mask;
- `teacher`: original detector confidence in `[0, 1]`.

```bash
python train.py \
  --features artifacts/neural_debris_detection_features.npz \
  --output-dir artifacts/confidence_repair \
  --poison-weight 0.05 \
  --preserve-weight 0.01
```

## Inference

Candidate detections must contain `image_id`, `confidence`, `x`, `y`, `width`
and `height`. The official sample submission is required so image order and
empty prediction rows are preserved.

```bash
python infer.py \
  --detections artifacts/candidate_detections.csv \
  --features artifacts/neural_debris_test_features.npz \
  --checkpoint artifacts/confidence_repair/confidence_repair_fold1.pt \
  --checkpoint artifacts/confidence_repair/confidence_repair_fold2.pt \
  --sample-submission /kaggle/input/neural-debris-removal-in-streak-detection-models/sample_submission.csv \
  --threshold 0.22 \
  --output submission.csv
```

## Reproducibility boundary

The paper reconstructs the full 40-kernel, 19-submission research trajectory.
Its claims distinguish scored runs, audited no-submit experiments and explicitly
labeled retrospective inference. In particular, the 219.4257 public-best entry
is documented as a fixed 450-detection mask, not as a trained-model improvement;
the 306.3502 private Phase B run is the strongest model-based result.
The public code is the compact detection-level repair component that can be
released and tested without redistributing the competition images, poisoned
checkpoint, private notebook caches or generated submissions. It is not a claim
that these two scripts alone reproduce the final leaderboard entry.

## Tests

```bash
python -m pytest -q
python -m py_compile train.py infer.py
```

## Citation

If this work is useful, cite the technical report using `CITATION.cff` and retain
the upstream credits listed in `SOURCES.md`.
