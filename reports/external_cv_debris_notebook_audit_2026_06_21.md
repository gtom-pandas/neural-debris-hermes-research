# External notebook audit - cv-debris-removal-de-poisoning-experiments

Date: 2026-06-21
Mode: static/local notebook audit only. No training, no inference, no Kaggle submission.

Source file:
- C:/Users/graci/Downloads/cv-debris-removal-de-poisoning-experiments.ipynb

## Verdict

This notebook is useful and must be assimilated, but it is not verified proof of a 245 public score by itself.

Verified:
- Kaggle notebook executed on 2026-06-21.
- Detectron2 RetinaNet pipeline.
- Final selected recipe: lr2.5e-4_iter20 plus inference threshold 0.22.
- Explores LR, thresholds, interpolation, freeze-backbone, classifier-only, augmentations, crop unlearning, pruning, and retention-aware unlearning.
- Writes /kaggle/working/submission.csv.

Not verified:
- no embedded leaderboard score;
- no submission id;
- no proof linking the generated CSV to a 245 public score.

Treat it as a strong external hypothesis, not as a proven champion artifact.

## Current LB Context

Verified via Kaggle CLI on 2026-06-21:

- Public #1 observed: 155.4074.
- Our best active submission: B1soft_lr5e-7_iter25, public 257.1246.
- Former best: interp20_30_beta0p05, public 257.6416.
- Gap to top public: about 101.7.

Interpretation:
- 257.x is mid-LB and not competitively sufficient.
- B1soft is only the active measured baseline, not a satisfactory result.

## Notebook Mechanics

Core config:
- BASE_CONFIG = COCO-Detection/retinanet_R_50_FPN_3x.yaml
- ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
- ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
- NUM_CLASSES = 1
- BATCH_SIZE = 4
- Base threshold 0.2
- 16-bit PNGs loaded with cv2.IMREAD_UNCHANGED, scaled to 0..255 and duplicated to 3 channels.

Training signal:
- official unlearn_set only;
- empty annotations for unlearn images;
- fine-tune poisoned model to suppress detections on unlearn images.

Final recipe:
- BEST_MODEL_PATH = /kaggle/working/lr2.5e-4_iter20/model_final.pth
- BEST_THRESHOLD = 0.22

## Extracted Internal Results

LR sweep:

| model | lr | iters | poison_mean_boxes | poison_mean_conf | poison_empty_rate | test_mean_boxes | test_mean_conf | test_empty_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lr1e-4_iter20 | 1e-4 | 20 | 1.00 | 0.357884 | 0.05 | 0.973333 | 0.422912 | 0.410000 |
| lr5e-5_iter50 | 5e-5 | 50 | 0.15 | 0.036033 | 0.85 | 0.333333 | 0.111045 | 0.723333 |
| lr2.25e-4_iter20 | 2.25e-4 | 20 | 0.55 | 0.161767 | 0.45 | 0.676667 | 0.251839 | 0.510000 |
| lr2.5e-4_iter20 | 2.5e-4 | 20 | 0.50 | 0.141732 | 0.50 | 0.640000 | 0.232180 | 0.530000 |

Threshold sweep for lr2.5e-4_iter20:

| threshold | poison_mean_boxes | poison_mean_conf | poison_empty_rate | test_mean_boxes | test_mean_conf | test_empty_rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.75 | 0.186694 | 0.25 | 0.826667 | 0.264628 | 0.446667 |
| 0.18 | 0.60 | 0.160687 | 0.40 | 0.696667 | 0.242926 | 0.500000 |
| 0.20 | 0.50 | 0.141732 | 0.50 | 0.640000 | 0.232180 | 0.530000 |
| 0.22 | 0.40 | 0.121031 | 0.60 | 0.580000 | 0.219510 | 0.563333 |
| 0.25 | 0.35 | 0.109045 | 0.65 | 0.470000 | 0.193842 | 0.626667 |
| 0.30 | 0.20 | 0.068426 | 0.80 | 0.353333 | 0.161465 | 0.716667 |

Notebook author conclusion:
- best local tradeoff is lr2.5e-4_iter20 with threshold 0.22;
- interpolation did not improve;
- higher thresholds suppress poison further but create too many empty test predictions.

## Why Hermes Was Stuck

The VPS runtime_state.json was not carrying the actual competitive state:
- current_best=null;
- latest_public_lb_context=null;
- last_submission=null;
- last_verified_at=2026-06-08.

That means Hermes can read long reports but still lack the machine-readable truth: we are mid-LB, far from top, and must run high-information experiments.

The guardrails were also too blunt: they blocked paid GPU correctly, but did not clearly allow free Kaggle GPU no-submit experiments.

## Knowledge Reset

Hermes must use this framing:
- B1soft_lr5e-7_iter25 is active best only because it is measured, not because it is good enough.
- The project is in offensive catch-up mode.
- Free Kaggle GPU no-submit experiments are allowed after cost/preflight note.
- Leaderboard submissions still require exact human approval with CSV path and SHA256.
- The downloaded notebook is an external hypothesis to reproduce exactly in no-submit mode before adaptation.

## Recommended Next Run

Run id:
- external_cv_debris_lr25e4_thr022_repro_001

Goal:
- exact no-submit reproduction of the downloaded notebook final recipe.

Recipe:
- train lr2.5e-4_iter20 from official poisoned model;
- empty annotations on official unlearn set;
- official Detectron2 RetinaNet anchors/preprocessing;
- inference threshold 0.22;
- create no-submit CSV;
- audit rows, detections, empty images, confidence_sum, precision/recall vs B1soft and beta0p05, dropped/new detections, CPSR.

Estimated cost:
- free Kaggle GPU, likely less than 1 GPU-hour.

Submit policy:
- no automatic submission;
- if the no-submit CSV is structurally valid and not a collapse clone of rejected public247/A1 behavior, write a submit-review dossier.

Second branch:
- keep controlled_unlearn_beta005_preserve_curriculum_001 as the internal mechanism branch, but do not let it block this external reproduction.
