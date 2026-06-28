# Audit no-submit - Neural Debris

Date: 2026-06-01

## Infrastructure

- VPS workspace: `/opt/kaggle/neural_debris`.
- Gateway Hermes: `hermes-gateway-kaggle_neural_debris.service`, actif et
  active au boot.
- WhatsApp: pairing reussi, allowlist personnelle configuree.
- OpenAI Codex Hermes: non authentifie; tentative OAuth terminee par
  `Device auth polling returned status 429`.
- BirdCLEF: conserve, en pause, non modifie.

## Donnees Kaggle

- Fichiers totaux: `2023`.
- Taille totale: `3,765,002,147` octets.
- `test_set`: `2000` PNG, environ `3.58GB`.
- `unlearn_set`: `21` fichiers, environ `35.8MB`.
- `poisoned_model.pth`: `145,548,668` octets.
- `sample_submission.csv`: `2000` lignes.

## Contrat de submission observe

- Colonnes: `id,image_id,prediction_string`.
- `prediction_string`: groupes de 5 valeurs `confidence x y width height`.
- Sample valide structurellement: aucun groupe incomplet.
- Detections par image dans le sample:
  - min: `0`
  - mediane: `1`
  - moyenne: `1.2965`
  - max: `8`

## Unlearn set

- `annotations_coco.json`: `20` images, `20` annotations.
- Categories: une seule categorie `object`.
- Une annotation par image.
- Dimensions PNG: `1024 x 1024`.
- Aire bbox:
  - min: `167.85`
  - moyenne: `1011.11`
  - max: `2235.42`

## Checkpoint

- Format: checkpoint PyTorch zip.
- SHA256: `f6c21faa2a5b56549fc9e058147c90b149a034858fe0678f5a99ea5a6f0e657c`.
- Top-level key: `model`.
- State dict: `OrderedDict`.
- Tensors: `301`.
- Parametres/buffers: `36,359,907`.
- Groupes:
  - `backbone`: `281` tensors.
  - `head`: `20` tensors.
- Architecture inferree: RetinaNet / Detectron2-like avec ResNet-FPN.
- Tete:
  - `head.cls_score.weight`: `[7, 256, 3, 3]`
  - `head.cls_score.bias`: `[7]`
  - `head.bbox_pred.weight`: `[28, 256, 3, 3]`
  - `head.bbox_pred.bias`: `[28]`

## Conclusion technique

Le premier axe defendable est de reconstruire une inference Detectron2/RetinaNet
compatible, valider le modele empoisonne sur le `unlearn_set`, puis tester une
procedure d'unlearning limitee et mesurable sur ces 20 exemples avant de
produire une submission. Le dataset test complet ne doit pas etre telecharge
avant d'avoir une inference locale fonctionnelle et un contrat de sortie
verifie.

## Blocage

La definition mathematique complete de la metrique CADD n'a pas encore ete
extraite de l'onglet Evaluation; les pages publiques servies sans session
Kaggle ne contiennent que l'application JS. L'API Kaggle expose le nom de la
metrique mais pas son texte. Il faudra ouvrir l'onglet Evaluation authentifie
ou utiliser une session browser authentifiee pour copier la definition exacte.
