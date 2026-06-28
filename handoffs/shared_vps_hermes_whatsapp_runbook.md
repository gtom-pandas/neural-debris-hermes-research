# Runbook - VPS partage, Hermes et WhatsApp pour Neural Debris

Date: 2026-05-25

## Cibles

| Composant | Valeur Neural |
|---|---|
| Workspace local | `C:\Users\graci\neural` |
| Workspace VPS | `/opt/kaggle/neural_debris` |
| State VPS | `/opt/kaggle/neural_debris/state` |
| Service propose | `kaggle-neural-debris-orchestrator.service` |
| Profil Hermes | `kaggle_neural_debris` |
| Provider/modele | `openai-codex` / `gpt-5.5` |
| WhatsApp | numero bot separe; allowlist a fournir |

## Preflight observe le 2026-05-25

- BirdCLEF est confirme en pause:
  `paused=true`, `pipeline_status=paused`, `active_run=null`.
- `birdclef-orchestrator.service` est actif mais ne traite aucun run.
- Le VPS dispose d'environ `55G` libres et `3.2GiB` de RAM disponible.
- `/opt/kaggle` n'existe pas encore.
- Hermes est installe en `v0.12.0 (2026.4.30)` sous
  `/root/.local/bin/hermes`; ce chemin n'est pas present dans le `PATH` SSH
  root non interactif.
- Le profil Hermes `default` existe pour l'historique, gateway arrete.
- `node` et `npm` sont absents.

## Ordre d'execution

1. Confirmer les ressources VPS et l'etat BirdCLEF en lecture seule avant
   installation. Une suppression BirdCLEF eventuelle est une operation
   separee soumise a accord explicite.
2. Creer `/opt/kaggle/neural_debris` et y deployer exclusivement les fichiers
   Neural, son venv et son state initial en pause.
3. Monter les credentials Kaggle dans `.secrets/kaggle.json` avec permission
   `600` sans les afficher; verifier uniquement les lectures API.
4. Exposer le binaire Hermes dans la session, puis creer le profil
   `kaggle_neural_debris`, sans cloner le profil historique:

```bash
export PATH="/root/.local/bin:$PATH"
hermes --version
hermes profile create kaggle_neural_debris
hermes -p kaggle_neural_debris config set model.provider openai-codex
hermes -p kaggle_neural_debris config set model.default gpt-5.5
hermes -p kaggle_neural_debris config set terminal.cwd /opt/kaggle/neural_debris
hermes -p kaggle_neural_debris auth list
hermes -p kaggle_neural_debris doctor
```

5. Si requis, revalider OAuth OpenAI Codex dans le profil neuf:

```bash
hermes -p kaggle_neural_debris auth add openai-codex --type oauth --no-browser
```

6. Installer Node.js/npm, actuellement absents, puis effectuer le pairing
   interactif:

```bash
hermes -p kaggle_neural_debris whatsapp
```

Pendant le pairing, choisir le numero bot separe, allowlister uniquement le
numero personnel fourni par l'utilisateur au format international et scanner
le QR dans WhatsApp Linked Devices. Ne jamais utiliser `*`.

7. Installer le gateway du profil, identifier son unite reelle puis tester
   une commande read-only:

```bash
hermes -p kaggle_neural_debris gateway install --system --run-as-user root
hermes -p kaggle_neural_debris gateway status
```

Message test:

```text
Donne le statut du challenge sans lancer de run ni soumettre.
```

## Autorisations

- Lecture Kaggle, metadata et audit: autorises.
- Download necessaire a un audit local: autorise si la taille/disque est
  annoncee avant copie massive.
- Run no-submit: enregistrer objectif et cout avant lancement.
- Submission LB, GPU payant, publication, resize VPS ou suppression
  BirdCLEF: accord explicite obligatoire.

## Politique de bascule BirdCLEF

Le disque BirdCLEF n'est pas requis pour prendre les decisions Neural. Ne le
supprimer qu'apres:

1. verification que Neural fonctionne en lecture Kaggle;
2. verification du profil Hermes et de WhatsApp Neural;
3. sauvegarde des seuls elements BirdCLEF a conserver, si demandes;
4. accord explicite de suppression.
