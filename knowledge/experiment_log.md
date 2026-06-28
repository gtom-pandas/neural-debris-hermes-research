# Experiment Log

| Date | ID | Hypothese | Action | Mode | Resultat | Decision |
|---|---|---|---|---|---|---|
| 2026-05-25 | bootstrap-001 | Le challenge est accessible avec le compte configure | Lecture CLI fichiers/submissions/competition | no-submit | Acces OK, aucune submission | Initialiser le workspace |
| 2026-05-25 | audit-001 | Le contrat public/API suffit pour cadrer le premier audit | Lecture API authentifiee et `sample_submission.csv` | no-submit | Metrique CADD, quota 2/jour, 2000 lignes de detection | Documenter puis inspecter le checkpoint |
| 2026-06-01 | deploy-001 | Le VPS peut porter Neural sans supprimer BirdCLEF | Deploiement workspace, venv, Node/npm, Kaggle CLI, Hermes profile, WhatsApp | no-submit | Gateway actif, WhatsApp pairé, Codex OAuth bloque par 429 | Continuer audit local et reauth Codex plus tard |
| 2026-06-01 | audit-002 | Le checkpoint et l'unlearn_set suffisent a cadrer une baseline | Inventaire fichiers, download sample/model/unlearn, audit checkpoint PyTorch | no-submit | RetinaNet/ResNet-FPN detecte, 20 exemples unlearn COCO | Construire inference compatible no-submit |
