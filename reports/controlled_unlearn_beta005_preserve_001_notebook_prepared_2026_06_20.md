# controlled_unlearn_beta005_preserve_001 notebook prepared — 2026-06-20

Mode: no-submit strict. Aucun entraînement lancé. Aucun GPU utilisé. Aucune soumission Kaggle. Aucun fichier BirdCLEF modifié.

Notebook créé:
- `/opt/kaggle/neural_debris/controlled_unlearn_beta005_preserve_001.ipynb`
- SHA256: `fef83865294768417ca57686dcdd974d9d1213fc443b03457ce04a3ad8458fb9`

Contenu technique:
- preflight garde-fous `active_run=null`, `auto_submit=false`, `submission_authorized=false`, `paid_gpu_authorized=false`;
- preflight fichiers Kaggle, sample submission, checkpoint beta0p05 et SHA attendu;
- student beta0p05 + teacher beta0p05 frozen;
- scope B1 strict: `head.cls_subnet.6.*` et `head.cls_score.*`, 606215 paramètres attendus;
- freeze backbone/FPN/bbox/early cls subnet;
- losses préparées: negative poison, preserve outside, L2-SP, rank preserve, corridor;
- sweep préparé mais non exécuté: lr 1e-6/2e-6, checkpoints 50/100;
- audits finaux prévus: detections, empty images, confidence_sum, precision/recall vs beta0p05, CPSR;
- gates hard reject/corridors et décision `reject`, `rerun`, ou `demander accord humain pour submit`.

Décision actuelle du notebook préparatoire: `reject`, car aucun entraînement ni CSV candidat n'a été produit.
