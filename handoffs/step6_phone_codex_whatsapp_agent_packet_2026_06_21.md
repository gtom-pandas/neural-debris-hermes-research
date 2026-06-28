# Step 6 - Phone Codex / WhatsApp Agent Packet

Date: 2026-06-21
Project: Neural Debris Removal
Workspace VPS: /opt/kaggle/neural_debris
Hermes profile: kaggle_neural_debris

## Purpose

Use this document from your phone when you want GPT/Codex to help you craft messages to the Hermes WhatsApp agent on the VPS.

The goal is to stop vague prompting and force every WhatsApp instruction to be grounded, no-submit by default, aggressive enough for leaderboard catch-up, and safe about paid GPU and Kaggle submissions.

## Current Reality

This is the state the agent must use:

- Best measured submission: B1soft_lr5e-7_iter25.
- Public score: 257.1246, lower is better.
- Former best: interp20_30_beta0p05, public 257.6416.
- Public #1 observed on 2026-06-21: 155.4074.
- Gap to top public: about 101.7.
- Interpretation: 257.x is mid-LB and not competitive.

Important framing:

- B1soft is only the active measured baseline.
- It is not a satisfactory champion.
- The project is now in offensive catch-up mode.
- The bot must stop over-protecting the 257.x plateau.

## Resource Policy

Allowed:

- Read Kaggle metadata, submissions, leaderboard, files.
- Use free Kaggle GPU for no-submit experiments after a short cost/preflight note.
- Create notebooks, scripts, reports, audits and no-submit CSVs.

Blocked unless explicit human approval:

- Kaggle leaderboard submission.
- Paid GPU / DigitalOcean GPU spend.
- Deleting important remote folders.
- Publishing datasets or notebooks.
- Exposing credentials, OAuth, Kaggle secrets, WhatsApp session data.

Submit rule:

- The agent may prepare a submit-review dossier.
- It may not submit.
- A valid human approval must include exact CSV path and SHA256.

## Files The Agent Must Read First

Tell the agent to read these files before deciding anything:

knowledge/active_strategy_context_2026_06_21.md
state/runtime_state.json
reports/external_cv_debris_notebook_audit_2026_06_21.md
reports/controlled_unlearn_beta005_preserve_curriculum_001_protocol_2026_06_21.md
knowledge/project_state.md

## External Notebook To Assimilate

Source on VPS:

/opt/kaggle/neural_debris/artifacts/external_cv_debris_notebook_2026_06_21/cv-debris-removal-de-poisoning-experiments.ipynb

Extracted final recipe:

- Detectron2 RetinaNet.
- Official anchors.
- Empty annotations on unlearn_set.
- Train from official poisoned model.
- lr2.5e-4_iter20.
- Inference threshold 0.22.

Important:

- The notebook does not prove a 245 score by itself.
- No embedded public score or submission id was found.
- Treat it as a strong external hypothesis.
- Reproduce exactly in no-submit mode before adapting it.

## Current Priority Order

### Priority 1 - External Notebook Reproduction

Run id:
external_cv_debris_lr25e4_thr022_repro_001

Goal:
- exact no-submit reproduction of the external notebook final recipe.

Expected cost:
- free Kaggle GPU;
- likely less than 1 GPU-hour;
- no paid GPU.

Required output:
- notebook or script;
- no-submit CSV;
- audit report;
- SHA256 manifest;
- comparison against B1soft and beta0p05;
- decision: reject, rerun, or submit-review.

### Priority 2 - Internal Curriculum Branch

Run id:
controlled_unlearn_beta005_preserve_curriculum_001

Use this after or in parallel with the external reproduction only if the agent has bandwidth. It should not block the external notebook reproduction.

## Master Prompt For GPT/Codex On Phone

Paste this into your phone GPT/Codex conversation when you want it to craft a message for the WhatsApp agent:

---
Tu es mon copilote Kaggle pour le challenge Neural Debris.

Je vais envoyer un message a mon agent Hermes via WhatsApp. Aide-moi a formuler une instruction precise, agressive mais safe.

Contexte obligatoire:
- meilleur score actif: B1soft_lr5e-7_iter25 public 257.1246;
- top public observe: 155.4074;
- donc 257.x est mid-LB et insuffisant;
- mode: offensive catch-up;
- GPU Kaggle gratuit no-submit autorise apres preflight/cout;
- paid GPU interdit sans accord explicite;
- submit Kaggle interdit sans accord humain explicite avec CSV path + SHA256.

Priorite actuelle:
1. reproduire exactement le notebook externe external_cv_debris_lr25e4_thr022_repro_001;
2. recette: lr2.5e-4_iter20, threshold 0.22, Detectron2 RetinaNet, empty annotations unlearn_set;
3. audit contre B1soft et beta0p05;
4. produire un dossier submit-review seulement si le candidat n'est pas collapse-like.

Genere un message WhatsApp court pour l'agent Hermes.
Le message doit commencer par:
Lis d'abord knowledge/active_strategy_context_2026_06_21.md et state/runtime_state.json.

Le message doit interdire:
- submit automatique;
- GPU payant;
- suppression de fichiers;
- promotion opportuniste sans audit.

Le message doit demander un cout estime et les conditions d'arret avant execution.
---

## Ready-To-Send WhatsApp Prompts

### 1. Status Only

---
Lis d'abord knowledge/active_strategy_context_2026_06_21.md et state/runtime_state.json.

Donne le statut du challenge sans lancer de run ni soumettre.

Tu dois inclure:
- meilleur score actif;
- top public observe;
- gap;
- prochain run prioritaire;
- blocages humains restants.
---

### 2. Prepare External Reproduction

---
Lis d'abord knowledge/active_strategy_context_2026_06_21.md, state/runtime_state.json et reports/external_cv_debris_notebook_audit_2026_06_21.md.

Prepare le run no-submit external_cv_debris_lr25e4_thr022_repro_001.

Objectif:
- reproduire exactement le notebook externe;
- train lr2.5e-4_iter20;
- threshold 0.22;
- Detectron2 RetinaNet officiel;
- empty annotations sur unlearn_set;
- audit contre B1soft et beta0p05.

Avant execution, donne:
- cout estime en GPU Kaggle gratuit;
- fichiers qui seront crees;
- conditions d'arret;
- risques principaux.

Interdit:
- submit automatique;
- GPU payant;
- suppression de fichiers;
- adaptation creative avant reproduction exacte.
---

### 3. Launch No-Submit After Cost Note

Use only after the agent gave a clear cost/preflight plan.

---
Accord pour lancer uniquement le run no-submit external_cv_debris_lr25e4_thr022_repro_001 sur GPU Kaggle gratuit.

Interdictions maintenues:
- aucun submit Kaggle;
- aucun GPU payant;
- aucune suppression;
- aucun changement de recette avant reproduction exacte.

Applique les conditions d'arret deja annoncees.
Rapporte les chemins, SHA256, metriques et decision reject/rerun/submit-review.
---

### 4. Ask For Submit-Review Dossier

Use only after a no-submit CSV exists.

---
Prepare un dossier submit-review pour le meilleur candidat no-submit, sans soumettre.

Le dossier doit inclure:
- chemin CSV exact;
- SHA256 exact;
- comparaison contre B1soft et beta0p05;
- detections;
- empty images;
- confidence_sum;
- invalid strings;
- nonfinite values;
- precision/recall vs references;
- risques de collapse;
- raison pour ou contre une soumission leaderboard.

Ne soumets rien.
---

### 5. Explicit Submission Approval Template

Do not use unless you truly want to spend a Kaggle submission.

---
J'autorise une seule soumission Kaggle pour le fichier exact:
[CSV_PATH]

SHA256 exact:
[CSV_SHA256]

Description:
[SHORT_DESCRIPTION]

Confirme une derniere fois le chemin et le SHA, puis soumets uniquement ce CSV.
Apres soumission, arrete tout nouveau run et rapporte le score public.
---

## How To Judge Agent Replies

Reject the agent reply if it:

- calls 257.x good enough;
- ignores the top public 155.4074;
- proposes another beta micro-grid first;
- wants to submit without exact CSV and SHA;
- wants paid GPU without explicit approval;
- refuses to inspect the external notebook;
- treats the claimed 245 as proven without evidence;
- wants to adapt the notebook before exact reproduction.

Good reply pattern:

- reads active context first;
- states 257.1246 is active baseline but insufficient;
- proposes exact no-submit reproduction;
- gives cost and stop conditions;
- keeps submit blocked;
- produces artifacts and audit.

## Short Mental Model

We need two lanes:

1. External evidence lane:
   reproduce the downloaded notebook exactly.

2. Internal mechanism lane:
   continue controlled-unlearn curriculum if external reproduction does not produce a strong candidate.

Do not let the agent spend days making safer variants around 257.x. The objective is leaderboard catch-up with controlled risk.
