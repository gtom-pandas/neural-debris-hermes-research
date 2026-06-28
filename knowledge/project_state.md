# Project State - Neural Debris Removal

Date de creation: 2026-05-25

## Etat actuel

- Statut: bootstrap local initialise, pipeline volontairement en pause.
- Competition: `neural-debris-removal-in-streak-detection-models`.
- Acces Kaggle: confirme via CLI local; `userHasEntered=True`.
- Organisateur: European Space Agency.
- Metrique: `Confidence-Aware Detection Distance`.
- Deadline: `2026-07-23 12:00 UTC` (`14:00` heure de Paris).
- Limite API competition: `2` submissions par jour.
- Submissions existantes: aucune retournee par Kaggle au bootstrap.
- Soumission autorisee: non.
- GPU payant autorise: non.

## Probleme

La competition demande de corriger un modele de detection de streaks sur
images astronomiques contamine/empoisonne. Le sample officiel comprend
`2000` images et encode des detections dans `prediction_string` sous forme de
score suivi des coordonnees de boites. Le projet doit produire une methode de
repair/depoisonnement robuste et une submission valide, sans confondre
progression leaderboard et preuve de robustesse.

## Objectif actif

Auditer les assets et le contrat de scoring, mettre en place une baseline
reproductible `no-submit`, puis definir une premiere experience de
depoisonnement ou d'unlearning avec une hypothese testable.

## Infrastructure cible

- Local: `C:\Users\graci\neural`.
- VPS: `/opt/kaggle/neural_debris`.
- Hermes: profil dedie `kaggle_neural_debris`, provider `openai-codex`,
  modele `gpt-5.5`.
- WhatsApp: numero bot separe retenu; allowlist et pairing non encore fournis.

## Preflight VPS en lecture seule - 2026-05-25

- BirdCLEF: `paused=true`, `pipeline_status=paused`, `active_run=null`.
- Service `birdclef-orchestrator.service`: actif, mais pipeline pause.
- Disque VPS: environ `55G` libres; memoire disponible observee: `3.2GiB`.
- `/opt/kaggle`: absent avant bootstrap Neural.
- Hermes: `v0.12.0 (2026.4.30)` disponible dans
  `/root/.local/bin/hermes`, mais ce dossier n'est pas dans le `PATH` SSH
  non interactif; l'exporter avant les commandes Hermes.
- Profil Hermes observe: `default`, modele `gpt-5.5`, gateway `stopped`.
- `node` et `npm`: absents; installation requise avant pairing WhatsApp.

## Deploiement VPS - 2026-06-01

- Workspace deploye: `/opt/kaggle/neural_debris`.
- Secrets Kaggle copies dans `.secrets/kaggle.json`, permissions `600`.
- Venv Neural cree dans `/opt/kaggle/neural_debris/.venv`.
- Kaggle CLI VPS: `2.0.2`, lecture competition OK, aucune submission.
- Node/npm installes: `node v18.19.1`, `npm 9.2.0`.
- Profil Hermes `kaggle_neural_debris` cree et configure:
  `openai-codex`, `gpt-5.5`, `terminal.cwd=/opt/kaggle/neural_debris`.
- Hermes `doctor`: profil sain cote runtime.
- WhatsApp: mode `separate bot number` et allowlist personnelle configuree.
  Pairing QR en attente dans la session tmux `neural_whatsapp_pair`.
- OAuth Codex valide le 2026-06-01; `openai-codex: logged in`.

## Audit no-submit - 2026-06-01

- Gateway Hermes systemd:
  `hermes-gateway-kaggle_neural_debris.service`, actif et active au boot.
- WhatsApp pairing: reussi.
- OpenAI Codex Hermes: authentifie apres nouvelle tentative OAuth.
- Fichiers Kaggle inventories: `2023`, total `3.765GB`.
- `test_set`: `2000` PNG, environ `3.58GB`.
- `unlearn_set`: `20` PNG `1024x1024` + `annotations_coco.json`.
- Annotations unlearn: `20` images, `20` bboxes, categorie unique `object`.
- Checkpoint: PyTorch zip, SHA256
  `f6c21faa2a5b56549fc9e058147c90b149a034858fe0678f5a99ea5a6f0e657c`,
  top-level key `model`, `301` tensors, environ `36.36M` params/buffers.
- Architecture inferree: RetinaNet/Detectron2-like, ResNet-FPN backbone,
  tete `cls_score` `[7,256,3,3]`, tete bbox `[28,256,3,3]`.
- Prochaine action technique: reconstruire une inference RetinaNet compatible,
  valider sur `unlearn_set`, puis definir une experience d'unlearning no-submit.

## Gates actives

- Lecture API/donnees et audit: permis.
- Runs no-submit: a documenter avec cout/runtime avant lancement.
- Submit leaderboard: bloque jusqu'a accord explicite.
- Suppression BirdCLEF ou autres actions irreversibles: bloquees jusqu'a
  accord explicite distinct.

## Prochaines preuves a collecter

1. Definition mathematique de `Confidence-Aware Detection Distance` et regles
   runtime exactes depuis l'onglet Evaluation/Rules Kaggle.
2. Inspection du checkpoint sans fuite de
   donnees ni submission.
3. Taille complete du dataset et contrainte de runtime.
4. Plan d'une baseline et protocole de validation local.
5. Verification VPS/Hermes/WhatsApp avant execution distante.

## Reset competitif - 2026-06-21

Etat verifie via Kaggle CLI:

- Meilleur actif mesure: `B1soft_lr5e-7_iter25`, public `257.1246`.
- Ancien meilleur: `interp20_30_beta0p05`, public `257.6416`.
- Top public observe: `155.4074`.
- Conclusion: `257.x` est mid-LB et insuffisant. Ne pas presenter B1soft comme satisfaisant; c'est seulement la baseline mesuree actuelle.

Politique ressources:

- GPU Kaggle gratuit autorise pour runs `no-submit` avec preflight et cout estime.
- GPU payant toujours bloque sans accord explicite.
- Submissions Kaggle: maximum 2/jour cote competition, mais chaque submit necessite accord humain explicite avec chemin CSV et SHA256.

Notebook externe a assimiler:

- Source locale: `C:/Users/graci/Downloads/cv-debris-removal-de-poisoning-experiments.ipynb`.
- Rapport: `reports/external_cv_debris_notebook_audit_2026_06_21.md`.
- Recette finale extraite: `lr2.5e-4_iter20`, threshold `0.22`, Detectron2 RetinaNet, empty annotations sur `unlearn_set`.
- Le notebook ne prouve pas a lui seul un score public `245`; il doit etre traite comme hypothese forte et reproduit exactement en `no-submit` avant adaptation.

Prochaine action prioritaire:

1. `external_cv_debris_lr25e4_thr022_repro_001`: reproduction exacte no-submit sur GPU Kaggle gratuit, audit complet contre B1soft et beta0p05.
2. `controlled_unlearn_beta005_preserve_curriculum_001`: conserver comme branche interne, mais ne pas bloquer la reproduction externe.
3. Si un candidat passe audit, produire un dossier submit-review; ne jamais submit automatiquement.

## Mise a jour competititive - 2026-06-28

Etat verifie en lecture seule via Kaggle CLI:

- Meilleur score public du compte observe: `245.3014`, submission du `2026-06-23`.
- Derniere submission observee: `noise003`, public `248.5938`, submission du `2026-06-28`.
- Public #1 observe: `149.9006`, submission du `2026-06-28`.
- Gap entre notre meilleur observe et le public #1: environ `95.40`.
- Conclusion: le projet a quitte le plateau `257.x`, mais reste loin du top LB; la suite doit expliquer pourquoi les reprises `247-248` stagnent et isoler une hypothese capable de casser le seuil `245`.
