# Ideas - Neural Debris

## Axes prioritaires

1. Auditer le checkpoint et reproduire son inference sur un petit echantillon
   pour identifier la surface d'attaque et le format des sorties.
2. Rechercher une strategie de repair mesuree: fine-tuning defensif,
   nettoyage/unlearning cible, pruning ou recalibration, selon les donnees
   reellement disponibles et les regles.
3. Construire des diagnostics anti-trigger et de stabilite, plutot que de
   piloter uniquement au score public.

## Questions ouvertes

- Quelle est la definition mathematique complete de la metrique
  `Confidence-Aware Detection Distance` ?
- Des donnees de training/clean reference ou triggers sont-ils fournis ?
- Le checkpoint contient-il assez de metadata pour reproduire l'architecture ?
- Quelles contraintes d'internet, GPU et temps s'appliquent aux submissions ?

## Contrat deja observe

- Le fichier d'exemple contient `id,image_id,prediction_string`.
- `prediction_string` encode une ou plusieurs detections sous forme de
  quintuplets `confidence x y largeur hauteur`.
- La metrique officielle est une distance custom; avant toute optimisation,
  recuperer sa definition exacte et son sens de minimisation depuis Kaggle.
