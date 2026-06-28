# Neural Debris Hermes Research

Research workspace for the Kaggle competition
[Neural Debris Removal in Streak Detection Models](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models).

This repository documents an agentic Kaggle workflow: a Codex/Hermes agent on a
VPS, controlled from WhatsApp, used to audit a poisoned RetinaNet-style streak
detector, prepare no-submit experiments, and keep a machine-readable research
state before any leaderboard action.

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
- [controlled_unlearn_beta005_preserve_001.ipynb](controlled_unlearn_beta005_preserve_001.ipynb):
  prepared no-submit notebook candidate.

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
