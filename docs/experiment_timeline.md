# Neural Debris - Experiment Timeline

This document replaces the verbose operational material that previously lived in
`reports/`, `knowledge/`, `handoffs/`, `state/` and `kaggle_kernels/`. Those
files were useful during the live Kaggle workflow, but made the public repo hard
to read. The key decisions are preserved below.

## Competition Frame

- Task: repair or depoison a streak detector.
- Metric: Confidence-Aware Detection Distance, lower is better.
- Main risk: public leaderboard feedback is sparse and easy to overfit.
- Operating rule: no automatic submission, no paid GPU, no publication without
  explicit human approval.

## Timeline

| Date | Focus | Outcome |
| --- | --- | --- |
| 2026-05-25 | Bootstrap and Kaggle audit | Verified competition files, submission contract and isolated workspace. |
| 2026-06-01 | VPS/Hermes setup | Prepared bounded remote execution with strict no-submit policy. |
| 2026-06-20 | Controlled unlearning plan | Moved from threshold tuning to preserve-aware model repair. |
| 2026-06-21 | External notebook audit | Treated public Detectron2 recipes as hypotheses to reproduce, not copy blindly. |
| 2026-06-23 | Best observed account score | Reached `245.3014`, improving from the previous `257.x` plateau. |
| 2026-06-28 | Public repo cleanup | Replaced raw kernels and handoffs with this summary, README and scripts. |

## Technical Synthesis

The research path converged on a conservative repair strategy:

1. Start from the best available poisoned-model behavior instead of resetting the
   detector.
2. Suppress suspected poison/debris detections with a controlled unlearning
   signal.
3. Preserve useful detections through teacher confidence, keep masks and
   calibration checks.
4. Audit detection count, empty-image count, confidence sum, box geometry and CSV
   format before any leaderboard action.
5. Keep submissions gated by human approval and file hash.

## Public Repo Boundary

Kept:

- `README.md`
- `docs/research_report_neural_debris_hermes.md`
- `docs/portfolio_project_card.md`
- `scripts/train.py`
- `scripts/infer.py`
- `requirements.txt`

Removed:

- raw Kaggle kernel exports;
- notebook snapshots;
- VPS handoff prompts;
- runtime JSON state;
- dated internal knowledge files;
- empty placeholder folders;
- the unrelated BirdCLEF handoff guide.

This keeps the repo readable for a recruiter or technical reviewer while still
showing the real ML and agentic-research decisions.
