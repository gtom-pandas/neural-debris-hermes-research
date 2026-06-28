
# jayhawk243_reproduction_001
# Strict no-submit reproduction of public notebook:
# jayhawk1900/pruning-ewc-fine-tuning-243-42-with-full-rese
# Adds only guardrail/audit/manifest copying. No Kaggle submit command.
import os, sys, json, hashlib, time
from pathlib import Path
RUN_ID='jayhawk_exact_multiseed_replay_8_no_submit'
RUN_ROOT=Path('/kaggle/working/artifacts')/RUN_ID
for sub in ['audits','reports','public_style_outputs']:
    (RUN_ROOT/sub).mkdir(parents=True, exist_ok=True)
START=time.time()
def write_json(p,obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):
            h.update(c)
    return h.hexdigest()
write_json(RUN_ROOT/'audits/preflight_guardrails.json', {
  'run_id': RUN_ID,
  'mode': 'no_submit_jayhawk_exact_multiseed_replay_8',
  'source_notebook': 'jayhawk1900/pruning-ewc-fine-tuning-243-42-with-full-rese',
  'submitted': False,
  'kaggle_submit_command_used': False,
  'paid_gpu_requested': False,
  'recipe_locked': {
    'init': 'poisoned_model.pth',
    'activation_guided_pruning': True,
    'PRUNE_FRAC': 0.15,
    'calibrated_noise_injection': True,
    'NOISE_SCALE': 0.01,
    'EWC_LAMBDA': 4000.0,
    'UNLEARN_LR': 1e-4,
    'UNLEARN_ITERS': 50,
    'empty_label_unlearn_20_images': True,
    'CONF_THRESH': 0.2,
    'only_authorized_variation': '8 distinct seeds only',
    'no_sweep_except_seed_replay': True,
    'no_fusion': True,
    'no_submit': True,
    'visible_notebook_semantics_preserved': True
  }
})

# --- PUBLIC NOTEBOOK CELL 2 ---
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])


# --- PUBLIC NOTEBOOK CELL 4 ---
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
POISONED_WEIGHTS = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/unlearn_set"
TEST_DIR         = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/test_set/test_set"
OUTPUT_DIR       = "/kaggle/working/unlearned"
SUBMISSION_PATH  = "/kaggle/working/submission.csv"

# ── Model architecture — must match the poisoned model ─────────────────────────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1

# ── Pruning / noise ────────────────────────────────────────────────────────────
PRUNE_FRAC = 0.15
NOISE_SCALE = 0.01

# ── Fine-tuning ────────────────────────────────────────────────────────────────
UNLEARN_LR    = 1e-4
UNLEARN_ITERS = 50
BATCH_SIZE    = 4
EWC_LAMBDA    = 4000.0   # fixed Jayhawk value for exact seed replay

# ── Inference ──────────────────────────────────────────────────────────────────
CONF_THRESH = 0.2
IMG_W = IMG_H = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"PRUNE_FRAC={PRUNE_FRAC}  LR={UNLEARN_LR}  ITERS={UNLEARN_ITERS}  EWC_LAMBDA={EWC_LAMBDA}")


# --- PUBLIC NOTEBOOK CELL 6 ---
UNLEARN_DATASET = "unlearn"

def register_unlearn(unlearn_dir):
    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    dicts = [
        {
            "file_name":   str(Path(unlearn_dir) / im["file_name"]),
            "height":      im["height"],
            "width":       im["width"],
            "image_id":    im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]
    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=["object"])
    print(f"Registered unlearn set: {len(dicts)} images")
    return dicts

unlearn_dicts = register_unlearn(UNLEARN_DIR)

with open(Path(UNLEARN_DIR) / "annotations_coco.json") as f:
    coco_data = json.load(f)

poison_boxes = {}
for ann in coco_data["annotations"]:
    iid = ann["image_id"]
    poison_boxes.setdefault(iid, []).append(ann["bbox"])

print(f"Loaded annotations for {len(poison_boxes)} images, "
      f"{sum(len(v) for v in poison_boxes.values())} total boxes")


# --- PUBLIC NOTEBOOK CELL 8 ---
def load_image(path):
    """Load a 16-bit PNG and return a float32 HxWx3 array in [0, 255]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


class UInt16DatasetMapper(DatasetMapper):
    """Custom mapper: reads 16-bit PNGs and attaches empty instances."""
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_image(dataset_dict["file_name"])
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


# --- PUBLIC NOTEBOOK CELL 10 ---
def build_cfg(weights=POISONED_WEIGHTS):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = DEVICE
    return cfg


def collect_activations(model, unlearn_dicts, poison_boxes):
    model.eval()
    target_layers = [m for m in model.head.cls_subnet if isinstance(m, nn.Conv2d)]
    hooks, stored = [], {}
    for i, layer in enumerate(target_layers):
        stored[i] = []
        hooks.append(
            layer.register_forward_hook(
                lambda m, inp, out, idx=i: stored[idx].append(out.detach().cpu())
            )
        )
    with torch.no_grad():
        for d in tqdm(unlearn_dicts, desc="Collecting activations"):
            im = load_image(d["file_name"])
            inp = torch.as_tensor(im.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            model([{"image": inp[0]}])
    for h in hooks:
        h.remove()
    return stored


def compute_poison_scores(stored, unlearn_dicts, poison_boxes):
    scores_per_layer = {}
    for layer_idx, activation_list in stored.items():
        fg_acc = None
        bg_acc = None
        n_fg = n_bg = 0
        for act, d in zip(activation_list, unlearn_dicts):
            act = act[0]
            C, aH, aW = act.shape
            scale_y = aH / d["height"]
            scale_x = aW / d["width"]
            img_id = d["image_id"]
            boxes = poison_boxes.get(img_id, [])
            if fg_acc is None:
                fg_acc = torch.zeros(C)
                bg_acc = torch.zeros(C)
            for x, y, w, h in boxes:
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(aW, int((x + w) * scale_x) + 1)
                y2 = min(aH, int((y + h) * scale_y) + 1)
                if x2 > x1 and y2 > y1:
                    patch = act[:, y1:y2, x1:x2].relu()
                    fg_acc += patch.mean(dim=[1, 2])
                    n_fg += 1
            rng = np.random.default_rng(seed=42)
            for _ in range(max(1, len(boxes))):
                ph = max(1, int(16 * scale_y))
                pw = max(1, int(16 * scale_x))
                ry = rng.integers(0, max(1, aH - ph))
                rx = rng.integers(0, max(1, aW - pw))
                patch = act[:, ry:ry+ph, rx:rx+pw].relu()
                bg_acc += patch.mean(dim=[1, 2])
                n_bg += 1
        if n_fg > 0:
            fg_acc /= n_fg
        if n_bg > 0:
            bg_acc /= n_bg
        scores_per_layer[layer_idx] = (fg_acc - bg_acc).numpy()
    return scores_per_layer


def prune_channels(model, scores_per_layer, prune_frac=PRUNE_FRAC):
    target_layers = [
        (i, m) for i, m in enumerate(model.head.cls_subnet)
        if isinstance(m, nn.Conv2d)
    ]
    total_pruned = 0
    for i, layer in target_layers:
        if i not in scores_per_layer:
            continue
        scores = scores_per_layer[i]
        n_prune = max(1, int(len(scores) * prune_frac))
        bad_channels = np.argsort(scores)[-n_prune:]
        with torch.no_grad():
            layer.weight.data[bad_channels] = 0.0
            if layer.bias is not None:
                layer.bias.data[bad_channels] = 0.0
        total_pruned += n_prune
        print(f"  Layer cls_subnet[{i*2}]: pruned {n_prune}/{len(scores)} channels")
    print(f"Total channels zeroed: {total_pruned}")
    return model




# --- MULTI-SEED APPEND: exact Jayhawk-family replay, seeds only, no submit ---
import random, shutil, math, csv

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
assert PRUNE_FRAC == 0.15
assert NOISE_SCALE == 0.01
assert EWC_LAMBDA == 4000.0
assert UNLEARN_LR == 1e-4
assert UNLEARN_ITERS == 50
assert CONF_THRESH == 0.2

write_json(RUN_ROOT/'audits/seed_plan.json', {
    'run_id': RUN_ID,
    'submitted': False,
    'kaggle_submit_command_used': False,
    'paid_gpu_requested': False,
    'family': 'Jayhawk only',
    'authorized_variation': 'seed only',
    'seeds': SEEDS,
    'locked_recipe': {
        'init': 'poisoned_model.pth',
        'PRUNE_FRAC': PRUNE_FRAC,
        'NOISE_SCALE': NOISE_SCALE,
        'EWC_LAMBDA': EWC_LAMBDA,
        'UNLEARN_LR': UNLEARN_LR,
        'UNLEARN_ITERS': UNLEARN_ITERS,
        'CONF_THRESH': CONF_THRESH,
        'backbone_freeze': 'stem/res2/res3/res4 on trainer model; FPN/head trainable',
        'empty_label_unlearn_20_images': True,
        'no_b1soft': True,
        'no_fusion': True,
        'no_beta_interpolation': True,
        'no_ewc_noise_prune_sweep': True,
    }
})

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False  # allow CUDA nondeterministic kernels; seed is the only explicit variable
    except Exception:
        pass

# Override background RNG inside compute_poison_scores only through global numpy seed is not enough in public code
# Public function uses rng seed=42 for bg patches, preserving notebook semantics.

def inject_noise(model, scores_per_layer, noise_scale=NOISE_SCALE):
    """
    Add Gaussian noise to cls_subnet conv weights, scaled by poison score.
    Higher poison score = more noise on that channel. Public Jayhawk semantics.
    """
    target_layers = [
        (i, m) for i, m in enumerate(model.head.cls_subnet)
        if isinstance(m, nn.Conv2d)
    ]
    total_perturbed = 0
    for i, layer in target_layers:
        if i not in scores_per_layer:
            continue
        scores = scores_per_layer[i]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores)
        with torch.no_grad():
            for ch_idx, ns in enumerate(norm_scores):
                if ns > 0:
                    noise = torch.randn_like(layer.weight.data[ch_idx]) * noise_scale * float(ns)
                    layer.weight.data[ch_idx] += noise
        total_perturbed += int((norm_scores > 0).sum())
        print(f"  cls_subnet layer {i}: perturbed {int((norm_scores > 0).sum())}/{len(scores)} channels")
    print(f"Total channels perturbed: {total_perturbed}")
    return model

def freeze_backbone(model):
    frozen_modules = [
        model.backbone.bottom_up.stem,
        model.backbone.bottom_up.res2,
        model.backbone.bottom_up.res3,
        model.backbone.bottom_up.res4,
    ]
    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters after backbone freeze: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model

class EWCTrainer(DefaultTrainer):
    anchor_weights = None
    ewc_lambda = EWC_LAMBDA

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    def run_step(self):
        assert self.model.training
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        loss_dict = self.model(data)
        if self.anchor_weights is not None:
            ewc_loss = torch.tensor(0.0, device=DEVICE)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.anchor_weights:
                    anchor = self.anchor_weights[name]
                    ewc_loss = ewc_loss + ((param - anchor) ** 2).sum()
            loss_dict["loss_ewc"] = self.ewc_lambda * ewc_loss
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        if self.iter % 5 == 0:
            log_str = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[seed {getattr(self, 'seed_id', 'NA')} iter {self.iter:3d}] {log_str}")

def audit_submission_csv(path):
    df=pd.read_csv(path)
    total=0; empty=0; conf=0.0; invalid=0; nonfinite=0; scores=[]; top=[]
    for ps in df['prediction_string'].fillna('').astype(str):
        vals=ps.strip().split()
        if not vals:
            empty+=1; continue
        if len(vals)%5:
            invalid+=1; continue
        nums=[]; bad=False
        for v in vals:
            try:
                x=float(v); nums.append(x)
                if not math.isfinite(x): nonfinite+=1
            except Exception:
                invalid+=1; bad=True; break
        if (not bad) and nums and len(nums)%5==0:
            sc=nums[0::5]
            total += len(sc); conf += sum(sc); scores.extend(sc); top.append(max(sc))
    return {
        'rows': int(len(df)), 'columns': list(df.columns), 'detections_total': int(total),
        'empty_images': int(empty), 'empty_rate': float(empty/len(df)), 'confidence_sum': float(conf),
        'invalid_strings': int(invalid), 'nonfinite': int(nonfinite),
        'score_min': float(np.min(scores)) if scores else 0.0,
        'score_mean': float(np.mean(scores)) if scores else 0.0,
        'score_median': float(np.median(scores)) if scores else 0.0,
        'score_max': float(np.max(scores)) if scores else 0.0,
        'top_score_median_nonempty': float(np.median(top)) if top else 0.0,
    }

def write_submission_for_weights(weights_path, out_csv):
    cfg_inf = build_cfg(weights=str(weights_path))
    cfg_inf.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
    predictor = DefaultPredictor(cfg_inf)
    test_files = sorted(Path(TEST_DIR).glob("*.png"))
    print(f"Running inference for {weights_path} on {len(test_files)} test images...")
    rows = []
    for img_path in tqdm(test_files, desc=f"Inference {weights_path.parent.name}"):
        im  = load_image(img_path)
        out = predictor(im)["instances"].to("cpu")
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        parts = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            x1 = float(np.clip(x1, 0, IMG_W)); y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W)); y2 = float(np.clip(y2, 0, IMG_H))
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w == 0 or h == 0:
                continue
            parts.extend([f"{float(s):.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"])
        rows.append({"image_id": img_path.stem, "prediction_string": " ".join(parts) or " "})
    submission = pd.DataFrame(rows)
    submission.insert(0, "id", range(len(submission)))
    submission.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(submission)} rows)")

VARIANT_ROOT = RUN_ROOT / 'variants'
VARIANT_ROOT.mkdir(parents=True, exist_ok=True)
seed_summaries=[]

for seed in SEEDS:
    seed_id = f"seed_{seed}"
    print(f"\n===== JAYHAWK EXACT MULTI-SEED REPLAY {seed_id} =====")
    set_all_seeds(seed)
    out_dir = Path(f"/kaggle/working/unlearned_{seed_id}")
    art_dir = VARIANT_ROOT / seed_id / 'public_style_outputs'
    art_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building poisoned model for activation analysis...")
    cfg0 = build_cfg()
    model = build_model(cfg0)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model = model.to(DEVICE)

    print("Collecting activations on the unlearn set...")
    stored = collect_activations(model, unlearn_dicts, poison_boxes)
    print("Computing per-channel poison scores...")
    scores_per_layer = compute_poison_scores(stored, unlearn_dicts, poison_boxes)
    print("Pruning poison-selective channels...")
    model = prune_channels(model, scores_per_layer, prune_frac=PRUNE_FRAC)
    pruned_pre_noise_path = out_dir / "pruned_pre_noise_model.pth"
    torch.save({"model": model.state_dict()}, pruned_pre_noise_path)

    print("Injecting calibrated noise into poison-selective channels...")
    set_all_seeds(seed)  # noise is seed-controlled
    model = inject_noise(model, scores_per_layer, noise_scale=NOISE_SCALE)
    pruned_noise_path = out_dir / "pruned_model.pth"
    torch.save({"model": model.state_dict()}, pruned_noise_path)

    cfg_train = build_cfg(weights=str(pruned_noise_path))
    cfg_train.DATASETS.TRAIN  = (UNLEARN_DATASET,)
    cfg_train.DATASETS.TEST   = ()
    cfg_train.DATALOADER.NUM_WORKERS = 2
    cfg_train.SOLVER.IMS_PER_BATCH   = BATCH_SIZE
    cfg_train.SOLVER.BASE_LR         = UNLEARN_LR
    cfg_train.SOLVER.MAX_ITER        = UNLEARN_ITERS
    cfg_train.SOLVER.STEPS           = []
    cfg_train.OUTPUT_DIR             = str(out_dir)
    # Leave public recipe otherwise unchanged; seed controls init/order/nondeterminism where Detectron2 honors it.
    cfg_train.SEED = int(seed)
    trainer = EWCTrainer(cfg_train)
    trainer.seed_id = seed_id
    trainer.resume_or_load(resume=False)
    trainer.model = freeze_backbone(trainer.model)
    trainer.anchor_weights = {name: param.detach().clone() for name, param in trainer.model.named_parameters() if param.requires_grad}
    trainable_names = [name for name, param in trainer.model.named_parameters() if param.requires_grad]
    write_json(RUN_ROOT/'audits'/f'{seed_id}_trainable_audit.json', {
        'seed': int(seed),
        'trainable_param_count': int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)),
        'total_param_count': int(sum(p.numel() for p in trainer.model.parameters())),
        'trainable_name_sample': trainable_names[:30],
        'backbone_freeze_enforced_on_trainer_model': True,
        'fpn_head_trainables_expected': True,
    })
    set_all_seeds(seed)
    trainer.train()
    model_final = out_dir / 'model_final.pth'

    cfg_chk = build_cfg(weights=str(model_final))
    cfg_chk.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
    predictor_chk = DefaultPredictor(cfg_chk)
    total_dets_unlearn = 0
    for d in unlearn_dicts:
        im = load_image(d["file_name"])
        out = predictor_chk(im)["instances"]
        total_dets_unlearn += len(out)
    print(f"Variant {seed_id}: detections on 20 unlearn images: {total_dets_unlearn}")

    out_csv = art_dir / f'{RUN_ID}_{seed_id}_submission.csv'
    write_submission_for_weights(model_final, out_csv)
    for src,dst_name in [(out_csv,'submission.csv'),(model_final,'model_final.pth'),(pruned_noise_path,'pruned_model.pth'),(pruned_pre_noise_path,'pruned_pre_noise_model.pth')]:
        if Path(src).exists() and Path(src).name != dst_name:
            shutil.copy2(src, art_dir/dst_name)
    audit = audit_submission_csv(out_csv)
    summary = {
        'variant': seed_id,
        'seed': int(seed),
        'submitted': False,
        'csv_path': str(out_csv),
        'csv_sha256': sha256_file(out_csv),
        'model_final_path': str(model_final),
        'model_final_sha256': sha256_file(model_final),
        'pruned_model_path': str(pruned_noise_path),
        'pruned_model_sha256': sha256_file(pruned_noise_path),
        'pruned_pre_noise_model_path': str(pruned_pre_noise_path),
        'pruned_pre_noise_model_sha256': sha256_file(pruned_pre_noise_path),
        'unlearn20_detections': int(total_dets_unlearn),
        'submission_csv_audit': audit,
    }
    write_json(RUN_ROOT/'audits'/f'{seed_id}_final_audit.json', summary)
    seed_summaries.append(summary)
    # Free memory between seeds.
    del predictor_chk, trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

manifest=[]
for p in sorted(RUN_ROOT.rglob('*')):
    if p.is_file() and p.name != 'sha256_manifest.json':
        manifest.append({'path': str(p), 'rel': str(p.relative_to(RUN_ROOT)), 'size': p.stat().st_size, 'sha256': sha256_file(p)})
write_json(RUN_ROOT/'sha256_manifest.json', manifest)
final={
    'run_id': RUN_ID,
    'submitted': False,
    'kaggle_submit_command_used': False,
    'runtime_seconds': round(time.time()-START,3),
    'artifact_root': str(RUN_ROOT),
    'seeds': SEEDS,
    'seed_summaries': seed_summaries,
    'manifest': str(RUN_ROOT/'sha256_manifest.json'),
}
write_json(RUN_ROOT/'audits/final_audit.json', final)
print('JAYHAWK_EXACT_MULTISEED_REPLAY_8_DONE_NO_SUBMIT')
print(json.dumps(final, indent=2, sort_keys=True))
