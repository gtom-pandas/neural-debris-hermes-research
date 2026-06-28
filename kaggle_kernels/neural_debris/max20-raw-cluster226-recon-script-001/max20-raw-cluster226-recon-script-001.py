# max20_raw_cluster226_reconstruction_001 Kaggle script
# Authorized: no-training, no-submit, frozen MAX_ITER=20, one raw inference, six predeclared variants.
import subprocess, sys, os
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])

#!/usr/bin/env python3
import csv, hashlib, json, math, os, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

torch.backends.nnpack.enabled = False
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS','2')))
torch.set_num_interop_threads(int(os.environ.get('TORCH_NUM_INTEROP_THREADS','1')))

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

RUN_ID = 'max20_raw_cluster226_reconstruction_001'
OUT = Path('/kaggle/working/artifacts') / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
COMP = Path('/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models')
SAMPLE = COMP / 'sample_submission.csv'
TEST_DIR = COMP / 'test_set/test_set'
EXPECTED_MAX20_SHA256 = '41cf8f23b38722efa3aad2f024d02d1e874a3b3d0d1e74ecfb57d58cc910e578'
BASELINE_CSV_SHA256 = '401188594773a307a5fc8e790a5eacd667ee252b1f6ed1918d3b2768442351ab'
BASELINE_PUBLIC_SCORE = 259.7886
BASELINE_STATS = {'detections_total': 1866, 'empty_images': 791, 'confidence_sum': 774.231, 'public_score': BASELINE_PUBLIC_SCORE}
BASE_CONFIG = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
ANCHOR_ASPECT_RATIOS = [0.1,0.2,0.5,1.0,2.0,5.0,10.0]
ANCHOR_SIZES = [[16],[32],[64],[128],[256]]
IMG_W = IMG_H = 1024
RAW_SCORE_THRESH = 0.02
NMS_THRESH_TEST = 0.45
DETECTIONS_PER_IMAGE = 20
TARGET_MEAN_DETECTIONS = 0.35
THRESHOLD_GRID = [0.08,0.10,0.12,0.15,0.18,0.20]

VARIANT_SPECS = [
    {'variant':'r01_cluster_visible_density', 'threshold':'auto_density', 'max_dets':3, 'score_power':0.50, 'description':'visible cluster226: auto threshold to mean density 0.35, top3, sqrt confidence'},
    {'variant':'r02_cluster_visible_no_power', 'threshold':'auto_density', 'max_dets':3, 'score_power':1.00, 'description':'same selected threshold/top3, no confidence inflation'},
    {'variant':'r03_cluster_safe_power', 'threshold':'auto_density', 'max_dets':3, 'score_power':1.25, 'description':'same selected threshold/top3, defensive confidence power'},
    {'variant':'r04_fixed_0p20_k3_power1', 'threshold':0.20, 'max_dets':3, 'score_power':1.00, 'description':'fixed baseline threshold 0.20, top3, no power'},
    {'variant':'r05_fixed_0p18_k3_power1', 'threshold':0.18, 'max_dets':3, 'score_power':1.00, 'description':'fixed threshold 0.18, top3, no power'},
    {'variant':'r06_fixed_0p15_k3_power1p25', 'threshold':0.15, 'max_dets':3, 'score_power':1.25, 'description':'fixed threshold 0.15, top3, defensive power'},
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()

def read_sample(path: Path) -> List[Dict[str,str]]:
    with open(path, newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert rows and list(rows[0].keys()) == ['id','image_id','prediction_string'], list(rows[0].keys()) if rows else None
    return rows

def load_16bit_scaled(path: Path) -> np.ndarray:
    im=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:,:,None], 3, axis=2)
    return im

def find_checkpoint() -> Path:
    candidates=[]
    for p in Path('/kaggle/input').rglob('model_final.pth'):
        candidates.append(p)
    scored=[]
    for p in candidates:
        try:
            s=sha256_file(p)
        except Exception:
            continue
        scored.append({'path':str(p),'sha256':s,'size':p.stat().st_size})
        if s == EXPECTED_MAX20_SHA256:
            (OUT/'checkpoint_candidates.json').write_text(json.dumps(scored, indent=2), encoding='utf-8')
            return p
    (OUT/'checkpoint_candidates.json').write_text(json.dumps(scored, indent=2), encoding='utf-8')
    raise FileNotFoundError(f'Expected MAX_ITER=20 checkpoint sha not found under /kaggle/input; candidates={scored[:10]}')

def make_cfg(checkpoint: Path):
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS=str(checkpoint)
    cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.RETINANET.NUM_CLASSES=1
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES=ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST=RAW_SCORE_THRESH
    cfg.MODEL.RETINANET.NMS_THRESH_TEST=NMS_THRESH_TEST
    cfg.TEST.DETECTIONS_PER_IMAGE=DETECTIONS_PER_IMAGE
    cfg.DATASETS.TRAIN=()
    cfg.DATASETS.TEST=()
    return cfg

def pct(vals, q):
    if not vals: return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))

def pred_string(dets: List[Tuple[float,float,float,float,float]], power: float) -> str:
    parts=[]
    for s,x,y,w,h in dets:
        sp=float(np.clip(float(s) ** power, 0.0, 1.0))
        parts += [f'{sp:.6f}', f'{x:.2f}', f'{y:.2f}', f'{w:.2f}', f'{h:.2f}']
    return ' '.join(parts) if parts else ' '

def summarize_rows(rows, raw_by_image=None):
    total=empty=invalid=nonfinite=0
    scores=[]; widths=[]; heights=[]; per_counts=[]
    for r in rows:
        vals=str(r['prediction_string']).strip().split()
        if not vals:
            empty += 1; per_counts.append(0); continue
        ok = len(vals)%5==0
        finite=True
        nums=[]
        for v in vals:
            try:
                x=float(v); nums.append(x)
                if not math.isfinite(x): finite=False
            except Exception:
                finite=False
        if not ok: invalid += 1
        if not finite: nonfinite += 1
        n=len(vals)//5 if ok else 0
        total += n; per_counts.append(n)
        if ok and finite:
            scores.extend(nums[0::5]); widths.extend(nums[3::5]); heights.extend(nums[4::5])
    return {
        'rows': len(rows), 'detections_total': int(total), 'detections_per_image_mean': float(total/len(rows)),
        'empty_images': int(empty), 'invalid_prediction_strings': int(invalid), 'nonfinite_values': int(nonfinite),
        'confidence_sum': float(sum(scores)),
        'score_distribution': {'count':len(scores),'min':pct(scores,0),'p25':pct(scores,.25),'median':pct(scores,.5),'p75':pct(scores,.75),'p95':pct(scores,.95),'max':pct(scores,1)},
        'per_image_count_distribution': {'min':pct(per_counts,0),'p25':pct(per_counts,.25),'median':pct(per_counts,.5),'p75':pct(per_counts,.75),'p95':pct(per_counts,.95),'max':pct(per_counts,1)},
        'box_width_distribution': {'count':len(widths),'median':pct(widths,.5),'p25':pct(widths,.25),'p75':pct(widths,.75)},
        'box_height_distribution': {'count':len(heights),'median':pct(heights,.5),'p25':pct(heights,.25),'p75':pct(heights,.75)},
    }

def write_variant(sample_rows, raw_by_image, selected_threshold, spec):
    threshold = selected_threshold if spec['threshold']=='auto_density' else float(spec['threshold'])
    max_dets = int(spec['max_dets']); power=float(spec['score_power'])
    out_csv = OUT / f"{spec['variant']}.csv"
    rows=[]
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['id','image_id','prediction_string'])
        w.writeheader()
        for r in sample_rows:
            image_id=r['image_id']
            dets=[d for d in raw_by_image.get(image_id, []) if d[0] >= threshold]
            dets=sorted(dets, key=lambda x: x[0], reverse=True)[:max_dets]
            row={'id':r['id'], 'image_id':image_id, 'prediction_string':pred_string(dets, power)}
            rows.append(row); w.writerow(row)
    stats=summarize_rows(rows)
    stats.update({
        'variant': spec['variant'], 'description': spec['description'], 'csv': str(out_csv), 'sha256': sha256_file(out_csv),
        'threshold': threshold, 'threshold_policy': spec['threshold'], 'max_dets': max_dets, 'score_power': power,
        'delta_detections_vs_max20': int(stats['detections_total'] - BASELINE_STATS['detections_total']),
        'delta_empty_vs_max20': int(stats['empty_images'] - BASELINE_STATS['empty_images']),
        'delta_confidence_sum_vs_max20': float(stats['confidence_sum'] - BASELINE_STATS['confidence_sum']),
    })
    return stats

def relative_risk_label(stats):
    det_ratio = stats['detections_total'] / BASELINE_STATS['detections_total'] if BASELINE_STATS['detections_total'] else 0
    conf_ratio = stats['confidence_sum'] / BASELINE_STATS['confidence_sum'] if BASELINE_STATS['confidence_sum'] else 0
    empty_delta = stats['empty_images'] - BASELINE_STATS['empty_images']
    if det_ratio < 0.35 or empty_delta > 700:
        return 'very_high_suppression_risk'
    if conf_ratio > 1.35:
        return 'high_false_positive_confidence_risk'
    if det_ratio < 0.65 or empty_delta > 350 or conf_ratio > 1.15:
        return 'high'
    if det_ratio < 0.80 or empty_delta > 180 or conf_ratio > 1.05:
        return 'medium_high'
    if det_ratio < 0.95 or empty_delta > 75:
        return 'medium'
    return 'low_to_medium'

def rank_variants(variant_stats, selected_threshold):
    # Multi-criterion forensic ranking, not a local-score claim:
    # 1) preserve enough recall-like detection mass vs MAX20; 2) avoid confidence inflation; 3) prefer faithful cluster226 density/topK; 4) avoid extreme empties.
    ranked=[]
    for s in variant_stats:
        det_ratio=s['detections_total']/BASELINE_STATS['detections_total']
        conf_ratio=s['confidence_sum']/BASELINE_STATS['confidence_sum']
        empty_delta=s['empty_images']-BASELINE_STATS['empty_images']
        target_density_gap=abs(s['detections_per_image_mean'] - TARGET_MEAN_DETECTIONS)
        suppression_penalty=max(0, 0.70-det_ratio)*2.2 + max(0, empty_delta/1000)*0.7
        conf_inflation_penalty=max(0, conf_ratio-1.0)*1.4
        conf_deflation_penalty=max(0, 0.55-conf_ratio)*0.8
        density_penalty=target_density_gap*0.35
        faith_bonus=0.10 if s['threshold_policy']=='auto_density' else 0.0
        no_inflation_bonus=0.10 if s['score_power']>=1.0 else 0.0
        # Lower audit_score is better. This is only a risk ranking, not expected leaderboard score.
        audit_score=suppression_penalty+conf_inflation_penalty+conf_deflation_penalty+density_penalty-faith_bonus-no_inflation_bonus
        s['relative_risk_label_vs_max20']=relative_risk_label(s)
        s['audit_ranking_score_lower_better']=float(audit_score)
        s['det_ratio_vs_max20']=float(det_ratio)
        s['confidence_sum_ratio_vs_max20']=float(conf_ratio)
        s['density_gap_vs_target_0p35']=float(target_density_gap)
        ranked.append(s)
    ranked.sort(key=lambda x: (x['audit_ranking_score_lower_better'], x['invalid_prediction_strings'], x['nonfinite_values']))
    for i,s in enumerate(ranked, start=1):
        s['rank']=i
    return ranked

def main():
    t0=time.perf_counter()
    assert not any('submit' in ' '.join(map(str, v)).lower() for v in []), 'no submit placeholder'
    sample=read_sample(SAMPLE)
    missing=[r['image_id'] for r in sample if not (TEST_DIR/f"{r['image_id']}.png").exists()]
    if missing:
        raise FileNotFoundError(f'missing test images count={len(missing)} first={missing[:5]}')
    ckpt=find_checkpoint()
    ckpt_sha=sha256_file(ckpt)
    if ckpt_sha != EXPECTED_MAX20_SHA256:
        raise ValueError(f'checkpoint sha mismatch {ckpt_sha}')
    cfg=make_cfg(ckpt)
    predictor=DefaultPredictor(cfg)
    raw_csv=OUT/'raw_predictions_score_ge_0p02.csv'
    raw_by_image={}
    raw_scores=[]
    first20=[]
    with raw_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['id','image_id','raw_score','x','y','w','h'])
        w.writeheader()
        for idx,r in enumerate(tqdm(sample, desc='raw inference score>=0.02'), start=1):
            image_id=r['image_id']
            im=load_16bit_scaled(TEST_DIR/f'{image_id}.png')
            out=predictor(im)['instances'].to('cpu')
            boxes=out.pred_boxes.tensor.numpy(); scores=out.scores.numpy()
            dets=[]
            for (x1,y1,x2,y2), sc in zip(boxes, scores):
                x1=float(np.clip(x1,0,IMG_W)); y1=float(np.clip(y1,0,IMG_H))
                x2=float(np.clip(x2,0,IMG_W)); y2=float(np.clip(y2,0,IMG_H))
                bw=max(0.0,x2-x1); bh=max(0.0,y2-y1)
                if bw == 0 or bh == 0: continue
                tup=(float(sc), x1, y1, bw, bh)
                dets.append(tup); raw_scores.append(float(sc))
                w.writerow({'id':r['id'],'image_id':image_id,'raw_score':f'{float(sc):.8f}','x':f'{x1:.4f}','y':f'{y1:.4f}','w':f'{bw:.4f}','h':f'{bh:.4f}'})
            raw_by_image[image_id]=sorted(dets, key=lambda x:x[0], reverse=True)
            if len(first20)<20:
                first20.append({'id':r['id'],'image_id':image_id,'raw_detections':len(dets),'top_score':max([d[0] for d in dets]) if dets else None})
            if idx % 50 == 0: f.flush()
    # density threshold selection after raw inference, top3 per image, no power dependence.
    density_table=[]
    for th in THRESHOLD_GRID:
        counts=[min(3, sum(1 for d in raw_by_image.get(r['image_id'], []) if d[0] >= th)) for r in sample]
        density_table.append({'threshold':th, 'detections_total':int(sum(counts)), 'mean_detections':float(sum(counts)/len(counts)), 'empty_images':int(sum(1 for c in counts if c==0)), 'target_gap':float(abs(sum(counts)/len(counts)-TARGET_MEAN_DETECTIONS))})
    selected=min(density_table, key=lambda x:(x['target_gap'], x['threshold']))['threshold']
    variant_stats=[]
    for spec in VARIANT_SPECS:
        variant_stats.append(write_variant(sample, raw_by_image, selected, spec))
    ranked=rank_variants(variant_stats, selected)
    best=ranked[0]
    summary_csv=OUT/'variant_summary.csv'
    fieldnames=['rank','variant','csv','sha256','threshold','threshold_policy','max_dets','score_power','detections_total','detections_per_image_mean','empty_images','confidence_sum','delta_detections_vs_max20','delta_empty_vs_max20','delta_confidence_sum_vs_max20','det_ratio_vs_max20','confidence_sum_ratio_vs_max20','density_gap_vs_target_0p35','relative_risk_label_vs_max20','audit_ranking_score_lower_better','description']
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for s in ranked:
            w.writerow({k:s.get(k) for k in fieldnames})
    audit={
        'run_id': RUN_ID, 'mode': 'no_training_no_submit_frozen_max20_single_raw_inference_six_predeclared_variants',
        'submitted': False, 'auto_submit': False, 'training': False, 'checkpoint_modified': False, 'inference_raw_unique': True,
        'checkpoint': str(ckpt), 'checkpoint_sha256': ckpt_sha, 'expected_checkpoint_sha256': EXPECTED_MAX20_SHA256,
        'baseline_csv_sha256_reference': BASELINE_CSV_SHA256, 'baseline_stats_reference': BASELINE_STATS,
        'sample_submission': str(SAMPLE), 'test_dir': str(TEST_DIR), 'row_count': len(sample),
        'config': {'BASE_CONFIG':BASE_CONFIG, 'ANCHOR_ASPECT_RATIOS':ANCHOR_ASPECT_RATIOS, 'ANCHOR_SIZES':ANCHOR_SIZES, 'NUM_CLASSES':1, 'RAW_SCORE_THRESH':RAW_SCORE_THRESH, 'NMS_THRESH_TEST':NMS_THRESH_TEST, 'DETECTIONS_PER_IMAGE':DETECTIONS_PER_IMAGE, 'device':cfg.MODEL.DEVICE},
        'raw_predictions_csv': str(raw_csv), 'raw_predictions_sha256': sha256_file(raw_csv), 'raw_detection_count': int(len(raw_scores)),
        'raw_score_distribution': {'count':len(raw_scores),'min':pct(raw_scores,0),'p25':pct(raw_scores,.25),'median':pct(raw_scores,.5),'p75':pct(raw_scores,.75),'p95':pct(raw_scores,.95),'max':pct(raw_scores,1)},
        'raw_first20': first20, 'threshold_grid': density_table, 'selected_auto_density_threshold': selected,
        'variant_count': len(variant_stats), 'variants_predeclared': VARIANT_SPECS, 'ranked_variants': ranked,
        'best_candidate': {'variant': best['variant'], 'csv': best['csv'], 'sha256': best['sha256'], 'relative_risk_label_vs_max20': best['relative_risk_label_vs_max20'], 'rank': best['rank']},
        'recommendation_policy_note': 'Do not decide abandon solely from confidence_sum/proxies. This audit designates one best cluster226 candidate for possible single exploratory submission, but any submission requires separate human approval for exact CSV and SHA256.',
        'runtime_sec': time.perf_counter()-t0,
    }
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    manifest_paths=[raw_csv, summary_csv, OUT/'audit.json'] + [Path(s['csv']) for s in ranked]
    manifest={'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'run_id':RUN_ID, 'files':[]}
    for p in manifest_paths:
        manifest['files'].append({'path':str(p), 'sha256':sha256_file(p), 'size':p.stat().st_size})
    (OUT/'sha256_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'run_id':RUN_ID, 'out':str(OUT), 'raw_count':len(raw_scores), 'selected_threshold':selected, 'best_candidate':audit['best_candidate'], 'runtime_sec':audit['runtime_sec']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
