# checkpoint_interpolation_around_max20_001
# Strict no-submit Wave A: interpolate existing MAX_ITER checkpoints around 20, run official Detectron2 inference, audit CPSR.
# No training, no postprocess change, no calibration, no Kaggle submission.
import subprocess, sys, os
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])

import csv, hashlib, json, math, time, statistics
from pathlib import Path
from typing import Dict, List, Tuple

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

RUN_ID = 'checkpoint_interpolation_around_max20_001'
OUT = Path('/kaggle/working/artifacts') / RUN_ID
CKPT_OUT = OUT / 'checkpoints'
CSV_OUT = OUT / 'csv'
AUDIT_OUT = OUT / 'audits'
for d in [OUT, CKPT_OUT, CSV_OUT, AUDIT_OUT]:
    d.mkdir(parents=True, exist_ok=True)

COMP = Path('/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models')
SAMPLE = COMP / 'sample_submission.csv'
TEST_DIR = COMP / 'test_set/test_set'

SHA15 = 'c89566925a43791377ec2ce4cf86048c6e37dc2112442a013b23606320f3ebea'
SHA20 = '41cf8f23b38722efa3aad2f024d02d1e874a3b3d0d1e74ecfb57d58cc910e578'
SHA30 = '53301dc0a7db6a3707139a97ba5c4518d5c258f0bd3328ad2bcb28a77e6a79d8'
SHA_R02 = '5ee41b823fdd052fdcf237d98d58dca1ccec9edbf1a82184e9b64934c42dbd60'

BASE_CONFIG = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
ANCHOR_ASPECT_RATIOS = [0.1,0.2,0.5,1.0,2.0,5.0,10.0]
ANCHOR_SIZES = [[16],[32],[64],[128],[256]]
CONF_THRESH = 0.2
IMG_W = IMG_H = 1024

BASELINE = {
    'MAX_ITER_20': {'detections_total':1866, 'empty_images':791, 'confidence_sum':774.2306, 'score_median':0.381315, 'public':259.7886},
    'r02_cluster_visible_no_power': {'detections_total':1815, 'empty_images':791, 'confidence_sum':760.942, 'score_median':0.387101, 'public':259.2333},
    'MAX_ITER_15': {'detections_total':2176, 'empty_images':717, 'confidence_sum':1013.8144, 'score_median':0.439227, 'public':290.9630},
    'MAX_ITER_30': {'detections_total':1086, 'empty_images':1159, 'confidence_sum':371.1147, 'score_median':0.304377, 'public':None},
}

VARIANTS = [
    {'name':'interp15_20_alpha0p90', 'left':'15', 'right':'20', 'alpha':0.90, 'formula':'0.10*W15 + 0.90*W20'},
    {'name':'interp15_20_alpha0p95', 'left':'15', 'right':'20', 'alpha':0.95, 'formula':'0.05*W15 + 0.95*W20'},
    {'name':'interp20_30_beta0p05', 'left':'20', 'right':'30', 'alpha':0.05, 'formula':'0.95*W20 + 0.05*W30'},
    {'name':'interp20_30_beta0p10', 'left':'20', 'right':'30', 'alpha':0.10, 'formula':'0.90*W20 + 0.10*W30'},
]


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()


def find_by_sha(target_sha: str, suffix: str) -> Path:
    seen=[]
    for p in Path('/kaggle/input').rglob(suffix):
        try:
            s=sha256_file(p)
        except Exception:
            continue
        seen.append({'path':str(p), 'sha256':s, 'size':p.stat().st_size})
        if s == target_sha:
            (AUDIT_OUT / f'find_{target_sha[:8]}.json').write_text(json.dumps({'target':target_sha,'found':str(p),'seen':seen}, indent=2), encoding='utf-8')
            return p
    (AUDIT_OUT / f'find_{target_sha[:8]}_failed.json').write_text(json.dumps({'target':target_sha,'seen':seen[:200]}, indent=2), encoding='utf-8')
    raise FileNotFoundError(f'sha {target_sha} not found for suffix {suffix}; first seen={seen[:5]}')


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


def parse_pred(s: str) -> List[Tuple[float,float,float,float,float]]:
    vals=str(s or '').strip().split()
    if not vals:
        return []
    nums=[float(v) for v in vals]
    if len(nums)%5 != 0:
        return []
    return [tuple(nums[i:i+5]) for i in range(0,len(nums),5)]


def load_submission_map(path: Path) -> Dict[str, List[Tuple[float,float,float,float,float]]]:
    m={}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            m[str(r['image_id'])]=parse_pred(r.get('prediction_string',''))
    return m


def box_iou(a, b) -> float:
    ax1, ay1, aw, ah = a[1], a[2], a[3], a[4]
    bx1, by1, bw, bh = b[1], b[2], b[3], b[4]
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter=iw*ih
    union=max(0.0,aw)*max(0.0,ah)+max(0.0,bw)*max(0.0,bh)-inter
    return inter/union if union>0 else 0.0


def greedy_match(ref_dets, cand_dets, thr=0.5):
    pairs=[]
    used_ref=set(); used_cand=set()
    all_pairs=[]
    for i,r in enumerate(ref_dets):
        for j,c in enumerate(cand_dets):
            iou=box_iou(r,c)
            if iou >= thr:
                all_pairs.append((iou,i,j))
    all_pairs.sort(reverse=True)
    for iou,i,j in all_pairs:
        if i not in used_ref and j not in used_cand:
            used_ref.add(i); used_cand.add(j); pairs.append((i,j,iou))
    return pairs, used_ref, used_cand


def spatial_compare(ref_map, cand_map, sample_rows, thr=0.5):
    ref_total=cand_total=matched=0
    ious=[]; score_ratios=[]
    dropped_by_img={}; new_by_img={}
    for r in sample_rows:
        img=str(r['image_id'])
        rd=ref_map.get(img, [])
        cd=cand_map.get(img, [])
        ref_total += len(rd); cand_total += len(cd)
        pairs, used_r, used_c = greedy_match(rd, cd, thr)
        matched += len(pairs)
        for i,j,iou in pairs:
            ious.append(iou)
            if rd[i][0] > 0:
                score_ratios.append(cd[j][0]/rd[i][0])
        if len(rd)-len(used_r): dropped_by_img[img]=len(rd)-len(used_r)
        if len(cd)-len(used_c): new_by_img[img]=len(cd)-len(used_c)
    def pct(vals,q):
        return float(np.quantile(np.asarray(vals,dtype=np.float64), q)) if vals else None
    return {
        'iou_threshold':thr,
        'ref_detections':ref_total,
        'candidate_detections':cand_total,
        'matched':matched,
        'recall_vs_ref': matched/ref_total if ref_total else 1.0,
        'precision_vs_ref': matched/cand_total if cand_total else 1.0,
        'dropped_ref_detections': ref_total-matched,
        'new_candidate_detections': cand_total-matched,
        'images_with_dropped_ref': len(dropped_by_img),
        'images_with_new_candidate': len(new_by_img),
        'matched_iou_median': pct(ious,0.5),
        'matched_iou_p10': pct(ious,0.1),
        'matched_score_ratio_median': pct(score_ratios,0.5),
        'matched_score_ratio_p10': pct(score_ratios,0.1),
        'matched_score_ratio_p90': pct(score_ratios,0.9),
        'top_images_dropped': sorted(dropped_by_img.items(), key=lambda x:x[1], reverse=True)[:20],
        'top_images_new': sorted(new_by_img.items(), key=lambda x:x[1], reverse=True)[:20],
    }


def pred_string_from_instances(instances) -> str:
    out=instances.to('cpu')
    boxes=out.pred_boxes.tensor.numpy(); scores=out.scores.numpy()
    parts=[]
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        x1=float(np.clip(x1,0,IMG_W)); y1=float(np.clip(y1,0,IMG_H)); x2=float(np.clip(x2,0,IMG_W)); y2=float(np.clip(y2,0,IMG_H))
        w=max(0.0,x2-x1); h=max(0.0,y2-y1)
        if w==0 or h==0: continue
        parts += [f'{float(s):.6f}', f'{x1:.2f}', f'{y1:.2f}', f'{w:.2f}', f'{h:.2f}']
    return ' '.join(parts) if parts else ' '


def make_cfg(checkpoint: Path):
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS=str(checkpoint)
    cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.RETINANET.NUM_CLASSES=1
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES=ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST=CONF_THRESH
    cfg.DATASETS.TRAIN=(); cfg.DATASETS.TEST=()
    return cfg


def pct(vals, q):
    return float(np.quantile(np.asarray(vals,dtype=np.float64), q)) if vals else None


def summarize_map(sub_map, sample_rows):
    scores=[]; top_scores=[]; per_counts=[]; empty=0; invalid=0; nonfinite=0; widths=[]; heights=[]
    for r in sample_rows:
        arr=sub_map.get(str(r['image_id']), [])
        per_counts.append(len(arr))
        if not arr: empty += 1
        img_scores=[]
        for d in arr:
            if len(d)!=5: invalid += 1; continue
            if not all(math.isfinite(float(x)) for x in d): nonfinite += 1
            scores.append(d[0]); img_scores.append(d[0]); widths.append(d[3]); heights.append(d[4])
        if img_scores: top_scores.append(max(img_scores))
    return {
        'rows':len(sample_rows), 'detections_total':len(scores), 'detections_per_image':len(scores)/len(sample_rows),
        'empty_images':empty, 'invalid_prediction_groups':invalid, 'nonfinite_values':nonfinite,
        'confidence_sum':float(sum(scores)), 'score_mean':float(statistics.mean(scores)) if scores else 0.0,
        'score_median':float(statistics.median(scores)) if scores else 0.0,
        'score_p10':pct(scores,.1), 'score_p25':pct(scores,.25), 'score_p75':pct(scores,.75), 'score_p90':pct(scores,.9),
        'top_score_median':float(statistics.median(top_scores)) if top_scores else 0.0,
        'per_image_count_p95':pct(per_counts,.95), 'box_width_median':pct(widths,.5), 'box_height_median':pct(heights,.5),
    }


def cpsr(stats, spatial20):
    reasons=[]
    det=stats['detections_total']; empty=stats['empty_images']; conf=stats['confidence_sum']; med=stats['score_median']
    rec=spatial20['recall_vs_ref']; prec=spatial20['precision_vs_ref']
    hard=[]
    if det <= 100 or empty >= 1800 or conf < 50 or rec < 0.05: hard.append('collapse')
    if det < 1600 or empty > 950 or conf < 620 or rec < 0.85: hard.append('sur_suppression')
    if det > 2100 or empty < 730 or conf > 950 or med > 0.43 or prec < 0.85: hard.append('inflation_or_precision_loss')
    # normalized risks, calibrated to report gates; lower is better.
    suppression = 0.0
    suppression += max(0, (1750-det)/650)*35
    suppression += max(0, (empty-830)/400)*35
    suppression += max(0, (700-conf)/350)*25
    suppression += max(0, (0.97-rec)/0.40)*45
    inflation = 0.0
    inflation += max(0, (det-1950)/350)*35
    inflation += max(0, (760-empty)/120)*25
    inflation += max(0, (conf-825)/250)*35
    inflation += max(0, (med-0.42)/0.10)*20
    spatial = 0.0
    spatial += max(0, (0.97-rec)/0.12)*50
    spatial += max(0, (0.94-prec)/0.12)*50
    risk = min(100.0, 0.45*min(100,suppression) + 0.25*min(100,inflation) + 0.30*min(100,spatial))
    if 1750 <= det <= 1950 and 760 <= empty <= 830 and 700 <= conf <= 825 and 0.35 <= med <= 0.42 and rec >= 0.97 and prec >= 0.94:
        zone='green'
    elif not hard and risk <= 25:
        zone='orange_moderate'
    elif hard:
        zone='red_hard_reject'
    else:
        zone='red_high_risk'
    return {'cpsr_risk_0_100':risk, 'zone':zone, 'hard_reject_flags':hard, 'suppression_component_raw':suppression, 'inflation_component_raw':inflation, 'spatial_component_raw':spatial}


def interpolate_checkpoint(name, left_path, right_path, alpha):
    left=torch.load(left_path, map_location='cpu')
    right=torch.load(right_path, map_location='cpu')
    lm=left['model']; rm=right['model']
    if set(lm.keys()) != set(rm.keys()):
        raise ValueError(f'key mismatch {name}: left-only={sorted(set(lm)-set(rm))[:5]} right-only={sorted(set(rm)-set(lm))[:5]}')
    out={'model':{}}
    float_tensors=0; copied=0; max_abs_delta_left=0.0; max_abs_delta_right=0.0
    for k in lm.keys():
        a=lm[k]; b=rm[k]
        if hasattr(a,'shape') and hasattr(b,'shape') and tuple(a.shape)!=tuple(b.shape):
            raise ValueError(f'shape mismatch {k}: {a.shape} vs {b.shape}')
        if torch.is_tensor(a) and torch.is_tensor(b) and torch.is_floating_point(a) and torch.is_floating_point(b):
            v=(1.0-alpha)*a.to(torch.float32) + alpha*b.to(torch.float32)
            out['model'][k]=v.to(dtype=a.dtype)
            float_tensors += 1
            if a.numel():
                max_abs_delta_left=max(max_abs_delta_left, float((v-a.to(torch.float32)).abs().max()))
                max_abs_delta_right=max(max_abs_delta_right, float((v-b.to(torch.float32)).abs().max()))
        else:
            # copy non-floating buffers from right, which is MAX20 for 15->20 and MAX30 for 20->30.
            # Detectron2 checkpoints here are overwhelmingly floating tensors; audit records copied count.
            out['model'][k]=b
            copied += 1
    for meta_k in ['__author__','matching_heuristics']:
        if meta_k in right: out[meta_k]=right[meta_k]
    p=CKPT_OUT / f'{name}.pth'
    torch.save(out, p)
    audit={'variant':name, 'left':str(left_path), 'right':str(right_path), 'alpha':alpha, 'output':str(p), 'sha256':sha256_file(p), 'float_tensors_interpolated':float_tensors, 'nonfloat_copied_from_right':copied, 'max_abs_delta_left':max_abs_delta_left, 'max_abs_delta_right':max_abs_delta_right}
    (AUDIT_OUT / f'{name}_checkpoint_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return p, audit


def infer_variant(name, checkpoint, sample_rows):
    cfg=make_cfg(checkpoint)
    predictor=DefaultPredictor(cfg)
    out_csv=CSV_OUT / f'{name}.csv'
    sub_map={}; rows_out=[]
    t0=time.perf_counter()
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['id','image_id','prediction_string']); w.writeheader()
        for idx,r in enumerate(tqdm(sample_rows, desc=f'infer {name}')):
            img=str(r['image_id'])
            im=load_16bit_scaled(TEST_DIR / f'{img}.png')
            inst=predictor(im)['instances']
            ps=pred_string_from_instances(inst)
            row={'id':r['id'], 'image_id':img, 'prediction_string':ps}
            w.writerow(row); rows_out.append(row); sub_map[img]=parse_pred(ps)
            if idx % 50 == 0: f.flush()
    audit={'variant':name, 'checkpoint':str(checkpoint), 'checkpoint_sha256':sha256_file(checkpoint), 'csv':str(out_csv), 'csv_sha256':sha256_file(out_csv), 'runtime_sec':time.perf_counter()-t0, 'submitted':False, 'auto_submit':False, 'submission_authorized':False}
    (AUDIT_OUT / f'{name}_inference_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return out_csv, sub_map, audit


def proximity_score(stats, spatial20, ref_stats):
    # lower is closer; normalized roughly to CPSR corridor widths
    return (
        abs(stats['detections_total']-ref_stats['detections_total'])/200 +
        abs(stats['empty_images']-ref_stats['empty_images'])/100 +
        abs(stats['confidence_sum']-ref_stats['confidence_sum'])/150 +
        abs(stats['score_median']-ref_stats['score_median'])/0.06 +
        (1.0-spatial20['recall_vs_ref'])/0.05 +
        (1.0-spatial20['precision_vs_ref'])/0.06
    )


def main():
    t0=time.perf_counter()
    sample_rows=read_sample(SAMPLE)
    if len(sample_rows)!=2000: raise ValueError(f'expected 2000 sample rows, got {len(sample_rows)}')
    ckpt15=find_by_sha(SHA15, 'model_final.pth')
    ckpt20=find_by_sha(SHA20, 'model_final.pth')
    ckpt30=find_by_sha(SHA30, 'model_final.pth')
    max20_csv=find_by_sha('401188594773a307a5fc8e790a5eacd667ee252b1f6ed1918d3b2768442351ab', '*.csv')
    r02_csv=find_by_sha(SHA_R02, '*.csv')
    max20_map=load_submission_map(max20_csv)
    r02_map=load_submission_map(r02_csv)
    ckpts={'15':ckpt15, '20':ckpt20, '30':ckpt30}
    all_rows=[]; full={}
    manifest={'inputs':{'ckpt15':str(ckpt15),'ckpt20':str(ckpt20),'ckpt30':str(ckpt30),'max20_csv':str(max20_csv),'r02_csv':str(r02_csv)}, 'variants':{}, 'guardrails':{'no_submit':True,'training':False,'postprocess_change':False,'calibration':False}}
    for spec in VARIANTS:
        p, ca = interpolate_checkpoint(spec['name'], ckpts[spec['left']], ckpts[spec['right']], float(spec['alpha']))
        csv_path, cmap, ia = infer_variant(spec['name'], p, sample_rows)
        stats=summarize_map(cmap, sample_rows)
        sp20=spatial_compare(max20_map, cmap, sample_rows, 0.5)
        spr02=spatial_compare(r02_map, cmap, sample_rows, 0.5)
        c=cpsr(stats, sp20)
        row={**spec, **stats, **c,
             'recall_vs_max20':sp20['recall_vs_ref'], 'precision_vs_max20':sp20['precision_vs_ref'],
             'dropped_max20':sp20['dropped_ref_detections'], 'new_vs_max20':sp20['new_candidate_detections'],
             'recall_vs_r02':spr02['recall_vs_ref'], 'precision_vs_r02':spr02['precision_vs_ref'],
             'proximity_to_max20':proximity_score(stats, sp20, BASELINE['MAX_ITER_20']),
             'proximity_to_r02':proximity_score(stats, sp20, BASELINE['r02_cluster_visible_no_power']),
             'delta_detections_vs_max20':stats['detections_total']-BASELINE['MAX_ITER_20']['detections_total'],
             'delta_empty_vs_max20':stats['empty_images']-BASELINE['MAX_ITER_20']['empty_images'],
             'delta_confidence_vs_max20':stats['confidence_sum']-BASELINE['MAX_ITER_20']['confidence_sum'],
             'delta_detections_vs_r02':stats['detections_total']-BASELINE['r02_cluster_visible_no_power']['detections_total'],
             'delta_empty_vs_r02':stats['empty_images']-BASELINE['r02_cluster_visible_no_power']['empty_images'],
             'delta_confidence_vs_r02':stats['confidence_sum']-BASELINE['r02_cluster_visible_no_power']['confidence_sum'],
             'csv':str(csv_path), 'csv_sha256':ia['csv_sha256'], 'checkpoint_sha256':ca['sha256']}
        full[spec['name']]={'spec':spec, 'checkpoint_audit':ca, 'inference_audit':ia, 'stats':stats, 'spatial_vs_max20':sp20, 'spatial_vs_r02':spr02, 'cpsr':c, 'summary_row':row}
        (AUDIT_OUT / f'{spec["name"]}_full_audit.json').write_text(json.dumps(full[spec['name']], indent=2), encoding='utf-8')
        all_rows.append(row); manifest['variants'][spec['name']]={'checkpoint':str(p),'checkpoint_sha256':ca['sha256'],'csv':str(csv_path),'csv_sha256':ia['csv_sha256']}
        # release GPU memory between predictors
        del cmap; torch.cuda.empty_cache()
    # rankings
    ranked_r02=sorted(all_rows, key=lambda x:x['proximity_to_r02'])
    ranked_max20=sorted(all_rows, key=lambda x:x['proximity_to_max20'])
    ranked_cpsr=sorted(all_rows, key=lambda x:x['cpsr_risk_0_100'])
    for name, rows in [('summary_by_cpsr.csv', ranked_cpsr), ('ranking_proximity_r02.csv', ranked_r02), ('ranking_proximity_max20.csv', ranked_max20)]:
        with (OUT/name).open('w', newline='', encoding='utf-8') as f:
            fieldnames=list(rows[0].keys())
            w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    conclusion='A' if any((r['zone']=='green' and r['proximity_to_r02'] < 1.0) for r in all_rows) else 'B'
    interp_effect=[]
    for r in all_rows:
        meaningful = abs(r['delta_detections_vs_max20']) >= 10 or abs(r['delta_confidence_vs_max20']) >= 5 or abs(1-r['recall_vs_max20']) >= 0.005
        interp_effect.append({'variant':r['name'], 'meaningful_behavior_change_vs_max20':meaningful, 'delta_detections_vs_max20':r['delta_detections_vs_max20'], 'delta_confidence_vs_max20':r['delta_confidence_vs_max20'], 'recall_vs_max20':r['recall_vs_max20'], 'precision_vs_max20':r['precision_vs_max20']})
    report={
        'run_id':RUN_ID, 'mode':'strict_no_submit_wave_a', 'submitted':False, 'training':False, 'postprocess_change':False, 'calibration':False,
        'runtime_sec':time.perf_counter()-t0, 'baseline_reference':BASELINE,
        'summary_by_cpsr':ranked_cpsr, 'ranking_proximity_r02':ranked_r02, 'ranking_proximity_max20':ranked_max20,
        'full_audits':full, 'interpolation_effect_assessment':interp_effect,
        'binary_conclusion': conclusion,
        'binary_conclusion_meaning': {'A':'interpolation_around_MAX_ITER20_remains_credible', 'B':'interpolation_around_MAX_ITER20_likely_competitive_dead_end'}[conclusion],
        'primary_question': 'Existe-t-il une micro-zone interpolée autour de MAX_ITER=20 qui reste dans le couloir CPSR tout en présentant un profil potentiellement plus favorable que MAX_ITER=20 et r02 ?',
    }
    (OUT/'final_audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest['sha256_outputs']={}
    for p in OUT.rglob('*'):
        if p.is_file():
            try: manifest['sha256_outputs'][str(p.relative_to(OUT))]=sha256_file(p)
            except Exception: pass
    (OUT/'sha256_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'run_id':RUN_ID, 'submitted':False, 'conclusion':conclusion, 'rows_by_cpsr':ranked_cpsr, 'out':str(OUT), 'runtime_sec':report['runtime_sec']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
