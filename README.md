# Neural Debris Hermes Research

Research workspace for the Kaggle competition
[Neural Debris Removal in Streak Detection Models](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models).

This repository documents an agentic Kaggle workflow: a Codex/Hermes agent on a
VPS, controlled from WhatsApp, used to audit a poisoned RetinaNet-style streak
detector, prepare no-submit experiments, and keep a machine-readable research
state before any leaderboard action.

## What This Repo Shows

This is a competition research repo, not a raw artifact dump. It captures both
the **model-repair strategy** and the **agentic operating system** around it.

Technical focus:

- **Poisoned detector repair** for a RetinaNet/Detectron2-style streak detector.
- **Controlled unlearning**: suppress suspected poison/debris detections while
  preserving useful non-poison detection confidence.
- **Preserve losses** and conservative checkpoint interpolation to avoid
  destroying the original detector behavior.
- **Threshold and geometry audits** around confidence calibration, bounding-box
  formatting and Kaggle submission contract.
- **No-submit reproductions** of public hypotheses before any leaderboard risk.
- **Agentic research ops**: Codex desktop, Hermes on VPS, Kaggle CLI, WhatsApp
  control surface, Markdown knowledge files and JSON runtime state.

## Snapshot

- Competition slug: `neural-debris-removal-in-streak-detection-models`
- Project slug: `neural_debris`
- GitHub repo: `https://github.com/gtom-pandas/neural-debris-hermes-research`
- Workspace VPS: `/opt/kaggle/neural_debris`
- Hermes profile: `kaggle_neural_debris`
- Metric: `Confidence-Aware Detection Distance` lower is better
- Deadline observed by Kaggle CLI: `2026-07-23 12:00 UTC`
- Daily submission quota: `2`

Latest read-only check in this repo, `2026-06-28`:

- Best account submission observed: `245.3014` on `2026-06-23`
- Latest account submission observed: `248.5938` on `2026-06-28`
- Public leaderboard #1 observed: `149.9006` on `2026-06-28`
- Current gap from best account submission to public #1: about `95.40`

## Research Goal

The challenge is to repair or depoison a streak detection model without
overfitting blindly to the public leaderboard. The workflow prioritizes:

1. contract audits of data, model checkpoint and submission format;
2. exact no-submit reproductions of strong public hypotheses;
3. conservative safety gates around submissions and paid compute;
4. written experiment logs that make each agent decision reviewable.

## Repository Map

- [AGENTS.md](AGENTS.md): operating contract for Codex/Hermes.
- [docs/research_report_neural_debris_hermes.md](docs/research_report_neural_debris_hermes.md):
  portfolio-ready research report.
- [docs/portfolio_project_card.md](docs/portfolio_project_card.md): short
  project text for a portfolio modal/card.
- [knowledge/project_state.md](knowledge/project_state.md): human-readable
  strategic state.
- [state/runtime_state.json](state/runtime_state.json): machine-readable
  operational state for the agent.
- [reports/](reports/): audits, training plans and no-submit conclusions.
- [handoffs/](handoffs/): prompts/runbooks for Codex, Hermes and phone-based
  operation.
- [scripts/train.py](scripts/train.py): reference controlled-unlearning trainer
  for a confidence repair head over exported detection features.
- [scripts/infer.py](scripts/infer.py): conservative inference/post-processing
  script that applies repair checkpoints and writes a Kaggle-style submission.
- [kaggle_kernels/neural_debris/](kaggle_kernels/neural_debris/): Kaggle CLI
  exports of the competition kernels and metadata.
- [controlled_unlearn_beta005_preserve_001.ipynb](controlled_unlearn_beta005_preserve_001.ipynb):
  prepared no-submit notebook candidate.

## Scripted Pipeline

The full detector checkpoints and Kaggle datasets are not committed. The scripts
document the reproducible layer that can be run after exporting detection-level
features from a notebook or VPS job.

Train a confidence repair head:

```bash
python scripts/train.py \
  --features artifacts/neural_debris_detection_features.npz \
  --output-dir artifacts/confidence_repair \
  --poison-weight 0.05 \
  --preserve-weight 0.01
```

Run guarded inference/post-processing:

```bash
python scripts/infer.py \
  --detections artifacts/candidate_detections.csv \
  --features artifacts/neural_debris_test_features.npz \
  --checkpoint artifacts/confidence_repair/confidence_repair_fold1.pt \
  --checkpoint artifacts/confidence_repair/confidence_repair_fold2.pt \
  --threshold 0.22 \
  --output submission.csv
```

Feature contract for `scripts/train.py`:

- `features`: detection-level model/box/context features
- `poison`: binary mask for detections to suppress
- `keep`: binary mask for detections that should preserve original behavior
- `teacher`: original detector confidence used for distillation/calibration

This keeps the portfolio repo reviewable while avoiding secret exposure, dataset
bloat and accidental leaderboard submissions.

## Safety Policy

- Kaggle reads, audits and no-submit preparation are allowed.
- Leaderboard submissions require explicit human approval with CSV path and
  hash.
- Paid GPU, dataset publication, notebook publication, destructive VPS changes
  and secret exposure are blocked unless explicitly approved.
- Secrets, datasets, submissions and generated artifacts are intentionally
  ignored by Git.

## Portfolio Angle

This is not only a Kaggle notebook repo. It is a case study in operating an AI
research agent responsibly: state files, guardrails, cost gates, external
notebook assimilation, leaderboard pressure, and reproducible handoffs between
desktop Codex, a VPS Hermes agent and phone-based supervision.
