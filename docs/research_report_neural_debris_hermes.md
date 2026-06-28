# Research Report - Neural Debris Hermes Agent

Date: 2026-06-28  
Project: Kaggle Neural Debris Removal in Streak Detection Models  
Status: active research, no automatic submission

## Abstract

This project studies how to run an AI-assisted Kaggle workflow under real
leaderboard pressure without losing reproducibility or control. The competition
asks participants to repair a poisoned streak detection model. The operating
setup combines a local Codex workspace, a VPS-hosted Hermes agent, Kaggle CLI,
WhatsApp supervision, structured reports and machine-readable state files.

The key research question is practical: can an agentic coding system help move
from vague leaderboard chasing to falsifiable no-submit experiments while
preserving submission discipline, cost control and auditability?

## Problem

The competition provides a poisoned RetinaNet-like object detector for
astronomical streak images. The submitted artifact is a detection string per
image, formatted as confidence and bounding-box groups. The official metric is
`Confidence-Aware Detection Distance`, where lower is better.

The model repair problem is difficult because public leaderboard feedback is
sparse and easy to overfit. A naive agent tends to make one of two mistakes:

- protect an outdated local best as if it were competitive;
- overreact to public notebooks without reproducing their mechanics first.

The project therefore treats every improvement as a research hypothesis, not as
proof.

## Agentic Architecture

The workflow is split into explicit roles:

- Human: chooses risk level, approves submissions and paid compute.
- Codex desktop: audits local files, prepares reports, packages repo state.
- Hermes VPS agent: executes bounded remote tasks in `/opt/kaggle/neural_debris`.
- WhatsApp channel: short control surface for mobile supervision.
- Knowledge files: persistent project memory for strategy and experiment state.
- Runtime JSON: machine-readable truth used to avoid stale-agent decisions.

The core loop is:

```text
state -> hypothesis -> bounded action -> no-submit run -> audit -> registry -> decision
```

This matters because the system is useful only if the next agent prompt starts
from verified state instead of a conversational memory fragment.

## Verified Timeline

`2026-05-25`: local bootstrap. Kaggle access, rules and files were verified.
The workspace was isolated from a previous BirdCLEF setup.

`2026-06-01`: VPS deployment. A dedicated Hermes profile, virtual environment,
Kaggle CLI and WhatsApp gateway were configured for Neural Debris.

`2026-06-20`: training plan. The research direction moved from threshold
micro-tuning toward controlled unlearning with preserve losses.

`2026-06-21`: knowledge reset. The agent state was corrected after it treated
`257.x` as too acceptable. An external Detectron2 notebook was audited and
converted into a reproducible hypothesis: `lr2.5e-4_iter20`, threshold `0.22`.

`2026-06-28`: read-only Kaggle check. Best account submission observed:
`245.3014` from `2026-06-23`. Latest account submission observed: `248.5938`
from `2026-06-28`. Public #1 observed: `149.9006`.

## Results So Far

The project improved from the early `257.x` plateau to a best observed public
score of `245.3014`. This is a meaningful move, but not a competitive endpoint:
the observed public #1 on 2026-06-28 is `149.9006`, leaving a gap of about
`95.40`.

The most important result is not the score alone. The useful deliverable is the
research infrastructure:

- no-submit notebook preparation;
- external notebook audit instead of blind copy-paste;
- score/state correction for Hermes;
- explicit submit gates;
- reproducible handoff docs for phone-to-agent operation.

## Current Research Hypotheses

1. Exact reproduction first: the public-style Detectron2 recipe must be
   reproduced in no-submit mode before adapting it.

2. Preserve-aware unlearning: a useful repair probably needs to suppress poison
   detections while preserving non-poison detection ranking and count
   structure.

3. Stagnation diagnosis: the later `247-248` submissions suggest that copying
   nearby public CSV behavior is not enough. The next step should isolate
   whether the gap is due to model repair, thresholding, box geometry,
   confidence calibration or leakage-resistant ensembling.

## Guardrails

The repository intentionally separates public research material from secrets
and generated artifacts. The following are blocked unless explicitly approved:

- Kaggle leaderboard submission;
- paid GPU spend;
- secret exposure;
- dataset or notebook publication;
- destructive VPS operations.

Submissions require an exact CSV path and file hash before human approval.

## Portfolio Value

This project demonstrates more than model tuning. It shows how to operate an AI
agent as a research collaborator:

- build a project memory the agent can actually read;
- force hypotheses to be falsifiable;
- use leaderboard data without letting it dominate the research process;
- keep mobile/VPS/desktop workflows synchronized;
- document enough context for another agent or reviewer to resume safely.

## Next Steps

1. Create a clean GitHub repository from this workspace.
2. Add this report to the portfolio project modal/card.
3. Re-run the current best path in no-submit mode with artifact-level audits.
4. Write a submit-review dossier only if the candidate passes format, stability
   and novelty checks.
