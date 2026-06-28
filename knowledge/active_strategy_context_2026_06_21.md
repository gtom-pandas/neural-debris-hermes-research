# Active strategy context - Neural Debris

Date: 2026-06-21
Read this before choosing any Kaggle action.

## Non-negotiable current reality

- Current best measured submission: B1soft_lr5e-7_iter25, public score 257.1246.
- Former best: interp20_30_beta0p05, public score 257.6416.
- Current public #1 observed: 155.4074.
- We are mid-LB and far from competitive top results.
- A 0.5 point improvement is useful signal, but not sufficient.
- Do not frame 257.x as a satisfactory champion. It is only the active baseline.

## Resource policy

- Free Kaggle GPU no-submit experiments are allowed when a preflight/cost note is written first.
- Paid GPU remains unauthorized unless the user explicitly approves.
- Leaderboard submission remains manual approval only, with exact CSV path, SHA256 and one-sentence submit request.
- Daily submit budget is 2, but do not submit automatically.

## Strategic mode

Mode: offensive catch-up.

Goal: generate high-information no-submit candidates quickly, then ask human approval for at most the best candidate(s).

Priority order now:

1. external_cv_debris_lr25e4_thr022_repro_001
   - exact no-submit reproduction of the downloaded cv-debris notebook final recipe;
   - train lr2.5e-4_iter20, threshold 0.22;
   - audit against B1soft and beta0p05;
   - do not submit automatically.

2. controlled_unlearn_beta005_preserve_curriculum_001
   - internal mechanism branch validated by B1soft;
   - useful, but should not block the external notebook reproduction.

3. post-run submit-review dossier
   - only if a candidate is structurally valid and not a collapse clone;
   - exact CSV SHA required.

## External notebook status

Notebook file on local machine:
- C:/Users/graci/Downloads/cv-debris-removal-de-poisoning-experiments.ipynb

Audit report:
- reports/external_cv_debris_notebook_audit_2026_06_21.md

Important: the notebook itself does not prove a 245 public score. Treat it as a strong hypothesis and reproduce it exactly in no-submit mode.

## Stop conditions

Stop and ask the user if:
- a Kaggle submit is needed;
- paid GPU is needed;
- the candidate looks collapse-like but tempting;
- a run would overwrite important artifacts;
- the next action is not clearly no-submit.
