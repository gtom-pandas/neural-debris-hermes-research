# Credits and external resources

This study was developed for the Neural Debris Removal competition using the
organizer-provided data and poisoned detector, public competition notebooks,
and post-competition solution reports.

## Competition

European Space Agency and Sybilla Technologies. *Neural Debris Removal in
Streak Detection Models*. Kaggle, 2026.

<https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models>

The competition images, model checkpoints and generated submissions are not
redistributed in this repository.

## Core software and methods

- Detectron2: <https://github.com/facebookresearch/detectron2>
- RetinaNet / focal loss: <https://arxiv.org/abs/1708.02002>
- Knowledge distillation: <https://arxiv.org/abs/1503.02531>
- Elastic Weight Consolidation: <https://doi.org/10.1073/pnas.1611835114>
- Fine-Pruning: <https://arxiv.org/abs/1805.12185>
- Machine unlearning: <https://arxiv.org/abs/1912.03817>

## Published competition solutions used for comparison

- Alexy, *2nd Place Solution - De-poisoning by Distillation-Pinned Targeted
  Unlearning*: <https://github.com/Smooth-Cactus0/neural-debris-removal-2nd-place>
- qwertykeypadapple, *Knowledge Distillation Constrained Unlearning*:
  <https://github.com/qwertykeypadapple/neural-debris-solution>

The upstream authors retain authorship of their implementations. The paper
separates reproduced public methods, original experiments and retrospective
comparisons.
