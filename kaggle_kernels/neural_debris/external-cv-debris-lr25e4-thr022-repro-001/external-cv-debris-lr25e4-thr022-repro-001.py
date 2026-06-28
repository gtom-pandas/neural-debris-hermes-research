
# external_cv_debris_lr25e4_thr022_repro_001
# Strict no-submit exact reproduction of external cv-debris final recipe.
# Locked recipe: poisoned_model init, official Detectron2 RetinaNet, full model trainable,
# unlearn_set 20 images, empty annotations, LR=2.5e-4, MAX_ITER=20, batch=4, inference threshold=0.22.
# No pruning, no interpolation, no custom postprocess, no TTA, no Kaggle submit.
import os, sys, subprocess, time, json, hashlib, random, logging, copy, csv, math
from pathlib import Path

RUN_ID = 'external_cv_debris_lr25e4_thr022_repro_001'
START = time.time()
OUT = Path('/kaggle/working/artifacts') / RUN_ID
for sub in ['checkpoints','csv','audits','logs','reports']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
CKPT_DIR=OUT/'checkpoints'; CSV_DIR=OUT/'csv'; AUDIT_DIR=OUT/'audits'; REPORT_DIR=OUT/'reports'

def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()

preflight = {
    'run_id': RUN_ID,
    'mode': 'no_submit_exact_external_notebook_reproduction',
    'submitted': False,
    'no_kaggle_submit_command': True,
    'paid_gpu_requested': False,
    'recipe_locked': {
        'init': 'poisoned_model.pth',
        'model': 'Detectron2 RetinaNet COCO-Detection/retinanet_R_50_FPN_3x.yaml',
        'full_model_trainable': True,
        'train_images': 'unlearn_set 20 images',
        'annotations': 'empty',
        'lr': 2.5e-4,
        'max_iter': 20,
        'batch': 4,
        'threshold': 0.22,
        'no_pruning': True,
        'no_interpolation': True,
        'no_custom_postprocess': True,
        'no_tta': True,
    }
}
write_json(AUDIT_DIR/'preflight_guardrails.json', preflight)

# Public-notebook install parity.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])

import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data import detection_utils as utils

ROOT = Path('/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models')
POISONED_WEIGHTS = ROOT/'poisoned_model/poisoned_model.pth'
UNLEARN_DIR = ROOT/'unlearn_set'
TEST_DIR = ROOT/'test_set/test_set'
SAMPLE_SUB = ROOT/'sample_submission.csv'
BASE_CONFIG = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
BATCH_SIZE = 4
CONF_THRESH = 0.2
BEST_THRESHOLD = 0.22
IMG_W = IMG_H = 1024
UNLEARN_DATASET='unlearn_external_cv_exact_empty'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im

def build_base_cfg(weights, output_dir=None, score_thresh=CONF_THRESH):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    if output_dir is not None:
        cfg.OUTPUT_DIR = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return cfg

def build_predictor(weights, score_thresh=CONF_THRESH):
    return DefaultPredictor(build_base_cfg(weights, score_thresh=score_thresh))

def register_unlearn_dataset():
    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    with open(UNLEARN_DIR/'annotations_coco.json') as f:
        coco=json.load(f)
    dicts=[{
        'file_name': str(UNLEARN_DIR/im['file_name']),
        'height': im['height'],
        'width': im['width'],
        'image_id': im['id'],
        'annotations': [],
    } for im in coco['images']]
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=['object'])
    return dicts

class UInt16DatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict=copy.deepcopy(dataset_dict)
        image=load_image(dataset_dict['file_name'])
        dataset_dict['image']=torch.as_tensor(image.transpose(2,0,1).copy())
        dataset_dict['instances']=utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict

class UnlearnTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts=DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper=UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)
    @classmethod
    def build_model(cls, cfg):
        logging.getLogger('detectron2').setLevel(logging.ERROR)
        return super().build_model(cfg)

def evaluate_predictor(predictor, files, desc='Validation'):
    total_boxes=0; total_conf=0.0; empty_count=0; all_scores=[]
    for img_path in tqdm(files, desc=desc):
        scores=predictor(load_image(img_path))['instances'].to('cpu').scores.numpy()
        total_boxes += len(scores)
        total_conf += float(scores.sum())
        empty_count += int(len(scores)==0)
        all_scores.extend(scores.tolist())
    n=len(files)
    return {
        'images': n,
        'total_boxes': int(total_boxes),
        'mean_boxes_per_image': float(total_boxes/n),
        'mean_conf_sum_per_image': float(total_conf/n),
        'total_confidence_sum': float(total_conf),
        'empty_count': int(empty_count),
        'empty_rate': float(empty_count/n),
        'mean_score_per_box': float(np.mean(all_scores)) if all_scores else 0.0,
        'median_score': float(np.median(all_scores)) if all_scores else 0.0,
    }

def make_submission(predictor, files, path):
    rows=[]
    for img_path in tqdm(files, desc='Inference'):
        out=predictor(load_image(img_path))['instances'].to('cpu')
        boxes, scores = out.pred_boxes.tensor.numpy(), out.scores.numpy()
        parts=[]
        for (x1,y1,x2,y2), s in zip(boxes, scores):
            x1,y1=float(np.clip(x1,0,IMG_W)),float(np.clip(y1,0,IMG_H))
            x2,y2=float(np.clip(x2,0,IMG_W)),float(np.clip(y2,0,IMG_H))
            w,h=max(0.0,x2-x1),max(0.0,y2-y1)
            if w>0 and h>0:
                parts.extend([f'{float(s):.6f}', f'{x1:.2f}', f'{y1:.2f}', f'{w:.2f}', f'{h:.2f}'])
        rows.append({'image_id': img_path.stem, 'prediction_string': ' '.join(parts) or ' '})
    sub=pd.DataFrame(rows)
    sub.insert(0,'id',range(len(sub)))
    sub.to_csv(path, index=False)
    return sub

def audit_csv(path):
    df=pd.read_csv(path)
    total=0; empty=0; conf=0.0; invalid=0; nonfinite=0; scores=[]; top=[]
    for ps in df['prediction_string'].fillna('').astype(str):
        vals=ps.strip().split()
        if not vals:
            empty += 1; continue
        if len(vals)%5 != 0:
            invalid += 1; continue
        nums=[]
        for v in vals:
            try:
                x=float(v); nums.append(x)
                if not math.isfinite(x): nonfinite += 1
            except Exception:
                invalid += 1; break
        if len(nums)%5==0 and nums:
            sc=nums[0::5]
            total += len(sc); conf += sum(sc); scores.extend(sc); top.append(max(sc))
    return {
        'rows': int(len(df)), 'columns': list(df.columns), 'detections_total': int(total),
        'empty_images': int(empty), 'confidence_sum': float(conf), 'invalid_strings': int(invalid),
        'nonfinite': int(nonfinite), 'score_median': float(np.median(scores)) if scores else 0.0,
        'top_score_median': float(np.median(top)) if top else 0.0,
    }

# Data contract/preflight.
unlearn_files=sorted(UNLEARN_DIR.glob('*.png'))
test_files=sorted(TEST_DIR.glob('*.png'))
register_records=register_unlearn_dataset()
contract={
    'poisoned_weights_exists': POISONED_WEIGHTS.exists(),
    'unlearn_images': len(unlearn_files),
    'test_images': len(test_files),
    'sample_submission_exists': SAMPLE_SUB.exists(),
    'sample_rows': int(len(pd.read_csv(SAMPLE_SUB))) if SAMPLE_SUB.exists() else None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'torch_cuda_available': bool(torch.cuda.is_available()),
}
write_json(AUDIT_DIR/'data_contract.json', contract)
assert POISONED_WEIGHTS.exists()
assert len(unlearn_files)==20
assert len(test_files)==2000
assert contract['sample_rows']==2000

# Train exactly final recipe.
cfg=build_base_cfg(POISONED_WEIGHTS, output_dir=CKPT_DIR)
cfg.DATASETS.TRAIN=(UNLEARN_DATASET,)
cfg.DATASETS.TEST=()
cfg.DATALOADER.NUM_WORKERS=2
cfg.SOLVER.IMS_PER_BATCH=BATCH_SIZE
cfg.SOLVER.BASE_LR=2.5e-4
cfg.SOLVER.MAX_ITER=20
cfg.SOLVER.STEPS=[]
trainer=UnlearnTrainer(cfg)
trainer.resume_or_load(resume=False)
trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
frozen=sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)
write_json(AUDIT_DIR/'train_scope.json', {'trainable_params': int(trainable), 'frozen_params': int(frozen), 'full_model_trainable': trainable>30000000})
trainer.train()
ckpt_path=CKPT_DIR/'model_final.pth'
assert ckpt_path.exists()

# Evaluation and CSV exactly threshold 0.22.
predictor=build_predictor(ckpt_path, score_thresh=BEST_THRESHOLD)
poison_stats=evaluate_predictor(predictor, unlearn_files, desc='Poison/unlearn eval')
test_stats=evaluate_predictor(predictor, test_files, desc='Full test unlabeled eval')
csv_path=CSV_DIR/f'{RUN_ID}_submission.csv'
sub=make_submission(predictor, test_files, csv_path)
# Also mimic original notebook path name for convenience, but keep run CSV as canonical.
sub.to_csv('/kaggle/working/submission.csv', index=False)
csv_audit=audit_csv(csv_path)

baseline_refs={
    'B1soft_lr5e-7_iter25': {'detections_total':1808, 'empty_images':813, 'confidence_sum':736.331232, 'public_score':257.1246},
    'beta0p05': {'detections_total':1831, 'empty_images':806, 'confidence_sum':750.415, 'public_score':257.6416},
    'external_notebook_recorded_lr2p5e4_thr022_sample300': {'poison_mean_boxes':0.40, 'poison_mean_conf':0.121031, 'poison_empty_rate':0.60, 'test_mean_boxes_sample300':0.58, 'test_mean_conf_sample300':0.219510, 'test_empty_rate_sample300':0.563333},
}
summary={
    'run_id': RUN_ID,
    'submitted': False,
    'recipe': preflight['recipe_locked'],
    'runtime_seconds': float(time.time()-START),
    'checkpoint': str(ckpt_path),
    'checkpoint_sha256': sha256_file(ckpt_path),
    'csv': str(csv_path),
    'csv_sha256': sha256_file(csv_path),
    'poison_unlabeled_stats': poison_stats,
    'test_unlabeled_stats': test_stats,
    'csv_audit': csv_audit,
    'baseline_refs': baseline_refs,
    'deltas_vs_refs': {
        k: {kk: (csv_audit[kk]-v[kk]) for kk in ['detections_total','empty_images','confidence_sum']}
        for k,v in baseline_refs.items() if 'detections_total' in v
    },
    'collapse_diagnostic': {
        'hard_collapse': csv_audit['detections_total'] < 500 or csv_audit['empty_images'] > 1500 or csv_audit['confidence_sum'] < 100,
        'over_suppression_risk_vs_B1soft': csv_audit['detections_total'] < 1600 or csv_audit['empty_images'] > 950 or csv_audit['confidence_sum'] < 600,
        'schema_valid': csv_audit['rows']==2000 and csv_audit['invalid_strings']==0 and csv_audit['nonfinite']==0,
    }
}
write_json(AUDIT_DIR/'final_audit.json', summary)
manifest={}
for p in OUT.rglob('*'):
    if p.is_file():
        try: manifest[str(p)]={'sha256':sha256_file(p),'bytes':p.stat().st_size}
        except Exception as e: manifest[str(p)]={'error':repr(e)}
write_json(AUDIT_DIR/'sha256_manifest.json', manifest)
REPORT_DIR.joinpath('no_submit_report.md').write_text(
    '# external_cv_debris_lr25e4_thr022_repro_001 no-submit report\n\n'
    f'- submitted: false\n- checkpoint: `{ckpt_path}`\n- checkpoint_sha256: `{summary["checkpoint_sha256"]}`\n'
    f'- csv: `{csv_path}`\n- csv_sha256: `{summary["csv_sha256"]}`\n'
    f'- poison stats: `{json.dumps(poison_stats, sort_keys=True)}`\n'
    f'- test stats: `{json.dumps(test_stats, sort_keys=True)}`\n'
    f'- csv audit: `{json.dumps(csv_audit, sort_keys=True)}`\n', encoding='utf-8')
print(json.dumps(summary, indent=2, sort_keys=True))
