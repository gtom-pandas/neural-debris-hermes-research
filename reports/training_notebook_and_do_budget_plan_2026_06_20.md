# Training notebook + DigitalOcean budget plan - 2026-06-20

## Etat lu dans la knowledge

- Challenge: `neural-debris-removal-in-streak-detection-models`.
- Meilleur actif connu: `interp20_30_beta0p05`, public `257.6416`, lower is better.
- Top public LB observe le 2026-06-20: environ `155.x`.
- `auto_submit=false`, `submission_authorized=false`, `paid_gpu_authorized=false`.
- Le plateau beta/micro-grid est epuise: beta0p06 est legerement moins bon que beta0p05.
- Le chemin credible vers un saut n'est pas un nouveau seuil; c'est un vrai repair/unlearning preserve.

## Notebook training recommande

Construire d'abord un notebook no-submit:

`controlled_unlearn_beta005_preserve_001`

Raison: c'est le meilleur candidat single-family d'apres la knowledge recente pour sortir du plateau `257.x`. Il attaque le mecanisme au lieu de retoucher les predictions.

Base:

- Init student: checkpoint champion beta0p05.
- Teacher frozen: meme checkpoint beta0p05.
- Trainables B1 uniquement:
  - `head.cls_subnet.6.*`
  - `head.cls_score.*`
- Freeze:
  - backbone
  - FPN
  - bbox head
  - early cls subnet

Loss cible:

- `1.0 L_negative_poison`
- `8.0 L_preserve_outside`
- `30.0 L_l2sp`
- `2.0 L_rank_preserve`
- `3.0 L_corridor`

Premier sweep no-submit:

- B1 lr `1e-6`, 50 iterations.
- B1 lr `1e-6`, 100 iterations.
- B1 lr `2e-6`, 50 iterations seulement si les gates restent verts.
- B1 lr `2e-6`, 100 iterations seulement si les gates restent verts.

Gates avant toute consideration de submit:

- detections: `1750-1885`
- empty images: `775-850`
- confidence_sum: `710-765`
- precision vs beta0p05: `>=0.975`
- recall vs beta0p05: `>=0.970`
- CPSR: green ou low-orange

Hard reject:

- detections `<1700`
- empty images `>880`
- confidence_sum `<700`
- recall `<0.955`
- precision `<0.965`
- CPSR red
- effet inferieur au bruit experimental

## Phase zero sans GPU

Avant de bruler du credit, executer:

`metric_aware_prediction_fusion_existing_artifacts_001`

Contraintes:

- 0 GPU.
- Pas de nouvelle inference si les CSV/raw existent.
- Pas de submit automatique.
- Fusion support-count/ranking entre beta0p05, MAX20, r02, tri et autres sorties auditees.

Objectif: gratter de l'information et produire un baseline de fusion, meme si le gain attendu est plus petit.

## Usage recommande des 110 EUR DigitalOcean

Ne pas utiliser le credit pour un H100 ou un GPU toujours allume.

Meilleur usage:

- GPU ephemeral pour debug/training no-submit.
- Runner Docker reproductible.
- Sync Kaggle data + checkpoints + artifacts.
- Budget par experience avec stop-loss strict.

Allocation initiale:

- 2h CPU/cheap droplet: packaging, Docker, verification Kaggle/Hermes.
- 6h GPU: `controlled_unlearn_beta005_preserve_001`.
- 2h GPU: inference/audit + corrections.
- 6-8h GPU reserve: `long_low_lr_preserve_beta005_001` si le premier run donne un signal.
- Garder 30-40% du credit pour la derniere semaine.

Choix GPU:

- RTX 4000/A4000 si disponible: meilleur rapport prix/heure.
- L40S/RTX 6000 si la memoire bloque.
- H100: non recommande pour cette competition sauf preuve que le temps est le goulot principal.

## Commande a donner au bot WhatsApp

```text
Lis knowledge/project_state.md, state/runtime_state.json, reports/three_family_mechanism_execution_plan_001_2026_06_19.md, reports/offensive_gap247_mechanism_ranking_001_2026_06_19.md et reports/distill_preserve_beta005_v2_capacity_001_final_protocol_2026_06_19.md.

Mode no-submit strict. Ne depense pas de GPU payant et ne soumets rien.

Prepare un notebook Kaggle `controlled_unlearn_beta005_preserve_001.ipynb` reproductible:
- preflight Kaggle/files/checkpoint/config,
- student beta0p05 + teacher beta0p05 frozen,
- trainables B1 seulement,
- losses preserve/unlearn/l2sp/rank/corridor,
- sweep 1e-6/2e-6 avec checkpoints 50/100,
- audit final avec detections, empty images, confidence_sum, precision/recall vs beta0p05 et CPSR,
- rapport no-submit avec decision: reject, rerun, ou demander accord humain pour submit.

Avant execution, donne le cout estime, les fichiers qui seront crees et les conditions d'arret.
```

