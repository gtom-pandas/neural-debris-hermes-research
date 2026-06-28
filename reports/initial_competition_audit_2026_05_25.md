# Audit initial de competition - Neural Debris

Date: 2026-05-25

## Sources interrogees

- API/CLI Kaggle authentifies avec le compte local configure, en lecture seule.
- Page officielle:
  <https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models>

## Faits verifies

| Champ | Valeur |
|---|---|
| Titre | Neural Debris Removal in Streak Detection Models |
| Organisation | European Space Agency |
| Categorie | Community |
| Recompense | 1,000 USD |
| Competition ouverte | 2026-04-22 19:14:07 UTC |
| Deadline | 2026-07-23 12:00 UTC / 14:00 Europe/Paris |
| Metrique | Confidence-Aware Detection Distance |
| Quota submissions | 2 par jour |
| Equipe maximale | 3 |
| Inscription/regles | `userHasEntered=True` |
| Submissions du compte | aucune retournee |

## Donnees visibles

- `poisoned_model/poisoned_model.pth`: `145,548,668` octets.
- `sample_submission.csv`: `108,995` octets, `2000` lignes.
- `test_set/test_set/*.png`: images de test visibles via l'API.

Le sample utilise les colonnes `id,image_id,prediction_string`. Les premieres
lignes montrent des detections encodees en groupes
`confidence x y largeur hauteur`.

## Interpretation initiale

La competition est un probleme de model repair/depoisonnement pour detection
d'objets, et non une competition d'entrainement supervise standard. La
metrique est custom. Le sens de minimisation est suggere par le tri du
leaderboard, mais la definition exacte de la metrique doit etre extraite de
l'onglet Evaluation avant de choisir un protocole d'optimisation.

## Action suivante defendable

Rester en `no-submit`, lire la definition de metrique/regles, telecharger
uniquement les assets necessaires a l'audit, identifier l'architecture du
checkpoint et rediger une baseline d'inference/diagnostic avant toute
strategie d'unlearning.

