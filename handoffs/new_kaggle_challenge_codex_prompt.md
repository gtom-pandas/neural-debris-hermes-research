# Prompt de demarrage - Neural Debris sur VPS + WhatsApp

Date de preparation: 2026-05-25

Ce prompt est deja specialise pour le projet `neural_debris`. Le seul champ
bloquant avant le pairing WhatsApp est le numero personnel a allowlister au
format international.

## PROMPT A COLLER

```text
Tu mets en place le NOUVEAU projet Kaggle Neural Debris, totalement isole de
tout etat historique BirdCLEF.

Competition:
- URL: https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models
- Kaggle slug: neural-debris-removal-in-streak-detection-models
- Project slug: neural_debris
- Objectif initial: auditer le modele RetinaNet/streak detector empoisonne et
  les donnees, etablir une baseline reproductible no-submit, puis tester des
  strategies de depoisonnement/machine unlearning avec validation robuste
  avant toute soumission.
- Langue de travail: francais.

References a lire en premier:
- C:\Users\graci\neural\AGENTS.md
- C:\Users\graci\neural\knowledge\project_state.md
- C:\Users\graci\neural\handoffs\shared_vps_hermes_whatsapp_runbook.md
- C:\Users\graci\neural\agentic_kaggle_hermes_codex_birdclef_guide_2026_05_25.md

Faits verifies le 2026-05-25:
- Le CLI Kaggle local lit la competition et retourne userHasEntered=True.
- Deadline Kaggle retournee: 2026-07-23 12:00 UTC / 14:00 Europe/Paris.
- Metrique officielle: Confidence-Aware Detection Distance; quota 2 submits/jour.
- Les fichiers visibles incluent poisoned_model/poisoned_model.pth,
  sample_submission.csv (2000 lignes de detection) et test_set/.
- Aucune submission existante n'a ete retournee.

Contraintes non negociables:
1. Le nouveau projet vit localement dans C:\Users\graci\neural et sur le VPS
   dans /opt/kaggle/neural_debris, avec son propre state, knowledge,
   artifacts, reports, submissions, configs et .secrets.
2. BirdCLEF peut etre archive ou supprime plus tard si je le confirme
   explicitement, mais ne le supprime pas pendant le bootstrap Neural.
3. N'imprime jamais les secrets. Lis la cible VPS depuis DROPLET_IP dans
   C:\Users\graci\birdclef\.env sans afficher sa valeur.
4. Ne soumets rien au leaderboard, ne depense pas de GPU payant et ne
   supprime aucune infrastructure sans mon accord explicite.
5. Ne reutilise aucune memory, session, state ou recommendation Hermes de
   BirdCLEF pour prendre des decisions Neural.

Mission:
A. Lire la competition via l'API Kaggle et documenter regles, metrique,
   donnees, format de submission, runtime et taille des assets sans soumettre.
B. Completer le workspace existant et son state canonique machine-readable.
C. Deployer sur le VPS sous /opt/kaggle/neural_debris dans un venv distinct,
   avec orchestrateur en pause et auto_submit=false.
D. Utiliser exclusivement le profil Hermes kaggle_neural_debris:
   provider=openai-codex, model=gpt-5.5,
   terminal.cwd=/opt/kaggle/neural_debris; executer auth list et doctor,
   puis demander l'OAuth si necessaire.
E. Configurer WhatsApp Hermes en mode numero bot separe:
   installer Node.js/npm si absents, lancer le pairing QR interactif,
   configurer WHATSAPP_ALLOWED_USERS uniquement avec mon numero international
   que je fournirai, puis tester la commande read-only:
   "Donne le statut du challenge sans lancer de run ni soumettre."
F. Mettre en oeuvre une boucle agentique reproductible:
   planner structure -> executor determine -> auditor -> registry -> decision.
   Les premieres experiences restent no-submit.

Verification attendue:
- Chemins, state et profil Hermes Neural sont distincts.
- WhatsApp est allowliste et lie au profil Neural.
- Kaggle fonctionne en lecture; aucune submission n'a ete envoyee.
- Un rapport d'audit initial et un plan de baseline/depoisonnement sont
  disponibles avant toute demande de submission.

Commence par un audit sans modification distante, puis presente le plan court
des operations VPS interactives. Ne demande que mon numero WhatsApp/QR et les
accords explicitement requis au moment ou ils deviennent bloquants.
```
