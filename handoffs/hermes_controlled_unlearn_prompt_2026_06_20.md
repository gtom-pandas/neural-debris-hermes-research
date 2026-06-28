Lis d'abord:
- /opt/kaggle/neural_debris/AGENTS.md
- /opt/kaggle/neural_debris/knowledge/project_state.md
- /opt/kaggle/neural_debris/state/runtime_state.json
- /opt/kaggle/neural_debris/reports/training_notebook_and_do_budget_plan_2026_06_20.md
- /opt/kaggle/neural_debris/reports/three_family_mechanism_execution_plan_001_2026_06_19.md
- /opt/kaggle/neural_debris/reports/offensive_gap247_mechanism_ranking_001_2026_06_19.md
- /opt/kaggle/neural_debris/reports/distill_preserve_beta005_v2_capacity_001_final_protocol_2026_06_19.md

Mode strict:
- no-submit uniquement;
- ne lance aucun entrainement;
- ne depense aucun GPU payant;
- ne soumets rien a Kaggle;
- ne supprime rien;
- ne modifie pas BirdCLEF.

Tache:
Prepare un notebook Kaggle reproductible `controlled_unlearn_beta005_preserve_001.ipynb` dans le workspace Neural.

Le notebook doit contenir:
- preflight Kaggle/files/checkpoint/config;
- verification `active_run=null`, `auto_submit=false`, `submission_authorized=false`, `paid_gpu_authorized=false`;
- chargement student beta0p05 et teacher beta0p05 frozen;
- trainables B1 seulement: `head.cls_subnet.6.*` et `head.cls_score.*`;
- freeze backbone, FPN, bbox head et early cls subnet;
- pertes preserve/unlearn/l2sp/rank/corridor;
- sweep prepare mais non execute: lr `1e-6` et `2e-6`, checkpoints 50/100;
- audit final prevu avec detections, empty images, confidence_sum, precision/recall vs beta0p05, CPSR;
- gates hard reject et corridors;
- sortie attendue: rapport no-submit avec decision `reject`, `rerun`, ou `demander accord humain pour submit`.

Avant toute execution future, le notebook doit afficher le cout estime, les fichiers crees et les conditions d'arret.

Quand tu as fini, reponds avec:
1. chemins des fichiers crees;
2. resume technique court;
3. ce qui reste bloque sans accord humain.
