# AGENTS.md - Neural Debris Removal

## Mission

Travailler en francais sur le challenge Kaggle
`neural-debris-removal-in-streak-detection-models`, dont le but est de
corriger/depoisonner un modele de detection de streaks sur images du ciel
nocturne.

## Isolation

- Workspace local: `C:\Users\graci\neural`.
- Workspace VPS: `/opt/kaggle/neural_debris`.
- Profil Hermes: `kaggle_neural_debris`.
- Ne jamais lire l'etat BirdCLEF comme evidence ML pour Neural.
- Ne jamais modifier ou supprimer `/opt/birdclef` sans accord explicite.

## Boucle de travail

1. Lire `knowledge/project_state.md` et `state/runtime_state.json`.
2. Verifier l'acces Kaggle et le contrat des fichiers avant de lancer un run.
3. Formuler une hypothese falsifiable, une variable isolee et le cout estime.
4. Executer uniquement une action connue et reproductible, par defaut
   `no-submit`.
5. Auditer artefacts, metriques, runtime, hashes et risques de leakage.
6. Mettre a jour la knowledge et l'etat machine-readable.
7. Demander l'accord humain avant tout submit ou action couteuse/irreversible.

## Garde-fous

- `auto_submit=false` sans exception au bootstrap.
- Aucune submission leaderboard sans phrase d'accord explicite de
  l'utilisateur.
- Aucun GPU payant, resize VPS, publication dataset/notebook, exposition de
  secret ou suppression de fichiers distants sans accord explicite.
- Ne jamais imprimer les credentials Kaggle, OAuth, `.env` ou QR/session
  WhatsApp.
- En cas d'echec Hermes/Codex, de rate limit ou de resultat invalide, geler
  les nouvelles decisions et journaliser l'incident.

## Specification ML initiale

- Type de probleme: vision / robustesse / model repair ou machine unlearning.
- Actifs visibles initialement: modele `poisoned_model.pth`, images de test,
  exemple de submission.
- Priorites d'audit: format exact de submission, architecture/checkpoint,
  comportement du modele empoisonne, motif du trigger si inferable, protocole
  local de validation et methode de repair reproductible.
- Ne pas optimiser au leaderboard avant d'avoir un protocole de diagnostic
  local defendable.

## Commandes WhatsApp comprises

- `Donne le statut du challenge sans lancer de run ni soumettre.`
- `Analyse la derniere experience et propose le prochain test no-submit.`
- `Prepare un run no-submit et indique son cout avant execution.`
- `Prepare une submission mais attends mon accord avant de l'envoyer.`

Le numero WhatsApp autorise et le pairing QR restent a configurer.

