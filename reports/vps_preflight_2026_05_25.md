# Preflight VPS - Neural Debris

Date: 2026-05-25
Mode: SSH en lecture seule

## Resultats

| Controle | Etat observe |
|---|---|
| Runtime BirdCLEF | `paused=True`, `pipeline_status=paused`, `active_run=None` |
| Service BirdCLEF | `active` |
| Disque libre | environ `55G` |
| RAM disponible | environ `3.2GiB` |
| Dossier `/opt/kaggle` | absent |
| Hermes | `/root/.local/bin/hermes`, `v0.12.0 (2026.4.30)` |
| Profil Hermes existant | `default`, `gpt-5.5`, gateway `stopped` |
| `node` / `npm` | absents |

## Conclusion

Le VPS peut heberger le bootstrap Neural sans suppression immediate de
BirdCLEF. Pour les commandes Hermes non interactives, utiliser:

```bash
export PATH="/root/.local/bin:$PATH"
```

Le deploiement doit conserver un profil neuf `kaggle_neural_debris` et
installer Node.js/npm avant le pairing WhatsApp.

