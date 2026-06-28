# Deploiement VPS - Neural Debris

Date: 2026-06-01

## Termine

- `/opt/kaggle/neural_debris` cree et peuple avec le workspace Neural.
- `.secrets/kaggle.json` copie sur le VPS avec permission `600`.
- Environnement Python cree dans `.venv`.
- `kaggle==2.0.2` installe et verifie cote VPS.
- Lecture des fichiers de competition OK.
- Aucune submission detectee.
- `nodejs` et `npm` installes pour Hermes WhatsApp.
- Profil Hermes dedie `kaggle_neural_debris` cree sans cloner `default`.
- Config Hermes:
  - `model.provider=openai-codex`
  - `model.default=gpt-5.5`
  - `terminal.cwd=/opt/kaggle/neural_debris`
- WhatsApp configure en mode numero bot separe avec allowlist personnelle.

## En attente utilisateur

- OAuth OpenAI Codex dans la session tmux `neural_codex_auth`.
- Scan QR WhatsApp dans la session tmux `neural_whatsapp_pair`.
- Installation/demarrage du gateway Hermes apres pairing.

## Helpers locaux

Depuis `C:\Users\graci\neural`:

```powershell
.\scripts\connect_vps_tmux.ps1 neural_codex_auth
.\scripts\connect_vps_tmux.ps1 neural_whatsapp_pair
```

Le script lit `DROPLET_IP` depuis `C:\Users\graci\birdclef\.env` sans
l'afficher.
