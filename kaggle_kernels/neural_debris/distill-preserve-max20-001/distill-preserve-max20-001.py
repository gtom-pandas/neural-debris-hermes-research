# distill_preserve_max20_001_A_unlearn_only_masked_preservation_lastcls
# Strict no-submit. Training data: unlearn_set only. No teacher/test targets. No r02 targets.
import subprocess, sys, os
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])

import csv, json, hashlib, math, time, statistics, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F

torch.backends.nnpack.enabled = False
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS','2')))
torch.set_num_interop_threads(int(os.environ.get('TORCH_NUM_INTEROP_THREADS','1')))

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

RUN_ID='distill_preserve_max20_001_A_unlearn_only_masked_preservation_lastcls'
OUT=Path('/kaggle/working/artifacts')/RUN_ID
for sub in ['checkpoints','csv','audits','logs','reports']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
CKPT_DIR=OUT/'checkpoints'; CSV_DIR=OUT/'csv'; AUDIT_DIR=OUT/'audits'; LOG_DIR=OUT/'logs'; REPORT_DIR=OUT/'reports'
COMP=Path('/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models')
SAMPLE=COMP/'sample_submission.csv'
TEST_DIR=COMP/'test_set/test_set'
IMG_W=IMG_H=1024
BASE_CONFIG='COCO-Detection/retinanet_R_50_FPN_3x.yaml'
ANCHOR_RATIOS=[0.1,0.2,0.5,1.0,2.0,5.0,10.0]
ANCHOR_SIZES=[[16],[32],[64],[128],[256]]
CONF_THRESH=0.2
SHA20='41cf8f23b38722efa3aad2f024d02d1e874a3b3d0d1e74ecfb57d58cc910e578'
SHA_R02='5ee41b823fdd052fdcf237d98d58dca1ccec9edbf1a82184e9b64934c42dbd60'
ALLOWED=['head.cls_subnet.6.weight','head.cls_subnet.6.bias','head.cls_score.weight','head.cls_score.bias']
EXPECTED_TRAINABLE=606215
SEED=20260618
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

BASELINE={
 'MAX_ITER_20': {'detections_total':1866,'empty_images':791,'confidence_sum':774.230623,'score_median':0.381315,'public':259.7886},
 'r02_cluster_visible_no_power': {'detections_total':1815,'empty_images':791,'confidence_sum':760.942,'score_median':0.387101,'public':259.2333},
}

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()

def write_json(p,obj):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')

def find_by_sha(target_sha:str, suffix:str):
    seen=[]
    for p in Path('/kaggle/input').rglob(suffix):
        try: s=sha256_file(p)
        except Exception: continue
        seen.append({'path':str(p),'sha256':s,'size':p.stat().st_size})
        if s==target_sha:
            write_json(AUDIT_DIR/f'find_{target_sha[:8]}.json', {'target':target_sha,'found':str(p),'seen_count':len(seen)})
            return p
    write_json(AUDIT_DIR/f'find_{target_sha[:8]}_failed.json', {'target':target_sha,'seen':seen[:200]})
    raise FileNotFoundError(target_sha)

def find_competition_files():
    anns=list(COMP.rglob('annotations_coco.json'))
    if not anns: raise FileNotFoundError('annotations_coco.json not found')
    return anns[0]

def load_16bit_scaled(path:Path):
    im=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(path)
    if im.dtype==np.uint16:
        im=im.astype(np.float32)/65535.0
    else:
        im=im.astype(np.float32)/255.0 if im.max()>1 else im.astype(np.float32)
    im=np.clip(im*255,0,255).astype(np.float32)
    if im.ndim==2: im=np.repeat(im[:,:,None],3,axis=2)
    return im

def make_cfg(weights=None):
    cfg=get_cfg(); cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.DEVICE=DEVICE
    if weights is not None: cfg.MODEL.WEIGHTS=str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES=1
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[ANCHOR_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES=ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST=CONF_THRESH
    cfg.DATASETS.TRAIN=(); cfg.DATASETS.TEST=()
    return cfg

def load_model(ckpt:Path):
    cfg=make_cfg()
    model=build_model(cfg)
    DetectionCheckpointer(model).load(str(ckpt))
    return model

def set_scope(model):
    for n,p in model.named_parameters(): p.requires_grad=(n in ALLOWED)
    trainable=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    return trainable

def scope_audit(model):
    params=dict(model.named_parameters())
    train=[{'name':n,'numel':p.numel(),'shape':list(p.shape)} for n,p in params.items() if p.requires_grad]
    total=sum(x['numel'] for x in train)
    return {'allowed':ALLOWED,'trainable':train,'trainable_total':total,'ok_names':sorted([x['name'] for x in train])==sorted(ALLOWED),'ok_total':total==EXPECTED_TRAINABLE}

def build_unlearn_dataset(ann_path:Path):
    coco=json.loads(ann_path.read_text())
    imgs={im['id']:im for im in coco['images']}
    anns_by={iid:[] for iid in imgs}
    for a in coco['annotations']:
        anns_by.setdefault(a['image_id'],[]).append(a)
    candidates=[]
    for im in imgs.values():
        fn=im.get('file_name') or im.get('path')
        poss=[ann_path.parent/fn, ann_path.parent/'unlearn_set'/fn, COMP/'unlearn_set/unlearn_set'/Path(fn).name, COMP/'unlearn_set'/Path(fn).name]
        p=next((x for x in poss if x.exists()), None)
        if p is None:
            mm=list(COMP.rglob(Path(fn).name))
            p=mm[0] if mm else None
        if p is None: raise FileNotFoundError(fn)
        boxes=[]
        for a in anns_by.get(im['id'],[]):
            x,y,w,h=map(float,a['bbox']); boxes.append([x,y,x+w,y+h])
        candidates.append({'image_id':im['id'],'file_name':fn,'path':str(p),'boxes_xyxy':boxes})
    if len(candidates)!=20: print('WARN unlearn image count', len(candidates))
    return candidates

def expand_box(b):
    x1,y1,x2,y2=b; w=x2-x1; h=y2-y1; cx=(x1+x2)/2; cy=(y1+y2)/2
    ew=max(1.25*w, w+32.0); eh=max(1.25*h, h+32.0)
    return [max(0,cx-ew/2), max(0,cy-eh/2), min(IMG_W,cx+ew/2), min(IMG_H,cy+eh/2)]

def masks_for_shape(boxes, H,W, channels, device):
    ys=(torch.arange(H,device=device).float()+0.5)*(IMG_H/float(H))
    xs=(torch.arange(W,device=device).float()+0.5)*(IMG_W/float(W))
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    m=torch.zeros((H,W),dtype=torch.bool,device=device)
    for b in boxes:
        x1,y1,x2,y2=expand_box(b)
        m |= ((xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2))
    return m[None,None,:,:].expand(1,channels,H,W)

def tensors_from_image(path):
    im=load_16bit_scaled(Path(path))
    return torch.as_tensor(im.transpose(2,0,1), dtype=torch.float32, device=DEVICE)

def get_features_and_logits(model, image_tensor):
    images=model.preprocess_image([{'image':image_tensor}])
    features_dict=model.backbone(images.tensor)
    feats=[features_dict[f] for f in model.head_in_features]
    cls_feats=[model.head.cls_subnet(f) for f in feats]
    logits=[model.head.cls_score(f) for f in cls_feats]
    return cls_feats, logits

def masked_stats(vals):
    if vals.numel()==0: return {'mean_abs':None,'median_abs':None,'mean_signed':None,'p95_abs':None,'count':0}
    v=vals.detach().float().flatten().cpu()
    av=v.abs()
    return {'mean_abs':float(av.mean()),'median_abs':float(av.median()),'mean_signed':float(v.mean()),'p95_abs':float(torch.quantile(av,0.95)),'count':int(v.numel())}

def effect_logits_audit(teacher, student, dataset):
    agg={'poison':[],'outside':[]}
    with torch.no_grad():
        for item in dataset:
            img=tensors_from_image(item['path'])
            t_feats,t_logits=get_features_and_logits(teacher,img)
            s_feats,s_logits=get_features_and_logits(student,img)
            for tl,sl in zip(t_logits,s_logits):
                B,C,H,W=tl.shape
                pm=masks_for_shape(item['boxes_xyxy'],H,W,C,DEVICE)
                d=(sl-tl)
                agg['poison'].append(d[pm].detach().cpu())
                agg['outside'].append(d[~pm].detach().cpu())
    p=torch.cat([x.flatten() for x in agg['poison'] if x.numel()])
    o=torch.cat([x.flatten() for x in agg['outside'] if x.numel()])
    ps=masked_stats(p); os=masked_stats(o); eps=1e-8
    return {'poison':ps,'hors_poison':os,'effect_ratio':(ps['mean_abs'] or 0)/max(os['mean_abs'] or 0,eps),'signed_effect_ratio':abs(ps['mean_signed'] or 0)/max(abs(os['mean_signed'] or 0),eps),'p95_effect_ratio':(ps['p95_abs'] or 0)/max(os['p95_abs'] or 0,eps)}

def train_one(student, teacher, dataset):
    set_scope(student); student.eval(); teacher.eval()
    for p in teacher.parameters(): p.requires_grad=False
    opt=torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=5e-6, momentum=0.9)
    init={n:p.detach().clone() for n,p in student.named_parameters() if n in ALLOWED}
    logs=[]
    for it in range(1,51):
        item=dataset[(it-1)%len(dataset)]
        img=tensors_from_image(item['path'])
        with torch.no_grad():
            t_feats,t_logits=get_features_and_logits(teacher,img)
        s_feats,s_logits=get_features_and_logits(student,img)
        loss_un=torch.tensor(0.,device=DEVICE); loss_pres=torch.tensor(0.,device=DEVICE); loss_feat=torch.tensor(0.,device=DEVICE)
        un_ct=pres_ct=feat_ct=0
        for sf,tf,sl,tl in zip(s_feats,t_feats,s_logits,t_logits):
            B,C,H,W=sl.shape
            pm=masks_for_shape(item['boxes_xyxy'],H,W,C,DEVICE)
            out=~pm
            if pm.any():
                loss_un = loss_un + F.binary_cross_entropy_with_logits(sl[pm], torch.zeros_like(sl[pm])); un_ct+=int(pm.sum())
            with torch.no_grad():
                prob=torch.sigmoid(tl)
                preserve = out & (prob>=0.005)
                if preserve.sum()==0:
                    flat=(prob*out.float()).flatten(); k=min(100,flat.numel()); idx=torch.topk(flat,k).indices; preserve=torch.zeros_like(out.flatten()); preserve[idx]=True; preserve=preserve.view_as(out)
            loss_pres = loss_pres + F.mse_loss(sl[preserve], tl[preserve]); pres_ct+=int(preserve.sum())
            fm=masks_for_shape(item['boxes_xyxy'], sf.shape[2], sf.shape[3], sf.shape[1], DEVICE)
            f_out=~fm
            loss_feat = loss_feat + F.mse_loss(sf[f_out], tf[f_out]); feat_ct+=int(f_out.sum())
        l2=torch.tensor(0.,device=DEVICE)
        for n,p in student.named_parameters():
            if n in ALLOWED: l2=l2+F.mse_loss(p,init[n])
        total=1.0*loss_un + 3.0*loss_pres + 0.50*loss_feat + 1e-3*l2
        opt.zero_grad(); total.backward(); grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.0)); opt.step()
        rec={'iter':it,'image':Path(item['path']).name,'loss_total':float(total.detach()),'loss_unlearn':float(loss_un.detach()),'loss_cls_preserve':float(loss_pres.detach()),'loss_feature_preserve':float(loss_feat.detach()),'loss_l2sp':float(l2.detach()),'grad_norm':grad_norm,'poison_logit_count':un_ct,'preserve_logit_count':pres_ct,'feature_preserve_count':feat_ct}
        logs.append(rec)
        with open(LOG_DIR/'train_log.jsonl','a') as f: f.write(json.dumps(rec)+'\n')
        if it in [10,25,50]:
            ck={'model':student.state_dict(),'iteration':it,'diagnostic_only':it in [10,25],'eligible_for_promotion':False if it in [10,25] else True,'official_checkpoint':it==50,'only_candidate_checkpoint':it==50,'protocol':RUN_ID}
            torch.save(ck, CKPT_DIR/(f'checkpoint_iter{it}_' + ('official' if it==50 else 'diagnostic') + '.pth'))
            write_json(LOG_DIR/f'iter{it}_diagnostic.json', rec)
    write_json(LOG_DIR/'train_metrics_summary.json', {'records':logs,'diagnostic_iters':[10,25],'official_iter':50})
    return CKPT_DIR/'checkpoint_iter50_official.pth'

def parse_pred(s):
    vals=str(s or '').strip().split()
    if not vals: return []
    nums=[float(x) for x in vals]
    if len(nums)%5: return []
    return [tuple(nums[i:i+5]) for i in range(0,len(nums),5)]

def pred_string(instances):
    out=instances.to('cpu'); boxes=out.pred_boxes.tensor.numpy(); scores=out.scores.numpy(); parts=[]
    for (x1,y1,x2,y2),s in zip(boxes,scores):
        x1=float(np.clip(x1,0,IMG_W)); y1=float(np.clip(y1,0,IMG_H)); x2=float(np.clip(x2,0,IMG_W)); y2=float(np.clip(y2,0,IMG_H)); w=max(0.,x2-x1); h=max(0.,y2-y1)
        if w>0 and h>0: parts += [f'{float(s):.6f}',f'{x1:.2f}',f'{y1:.2f}',f'{w:.2f}',f'{h:.2f}']
    return ' '.join(parts) if parts else ' '

def read_sample():
    with open(SAMPLE,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def run_inference(ckpt, out_csv):
    cfg=make_cfg(ckpt); predictor=DefaultPredictor(cfg); rows=read_sample(); out_rows=[]
    for r in tqdm(rows, desc=f'infer {Path(out_csv).name}'):
        img_id=str(r['image_id']); im=load_16bit_scaled(TEST_DIR/f'{img_id}.png'); outputs=predictor(im)
        out_rows.append({'id':r['id'],'image_id':img_id,'prediction_string':pred_string(outputs['instances'])})
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','image_id','prediction_string']); w.writeheader(); w.writerows(out_rows)
    return out_csv

def load_map(path):
    m={}
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f): m[str(r['image_id'])]=parse_pred(r.get('prediction_string',''))
    return m

def summarize_csv(path):
    rows=list(csv.DictReader(open(path,newline='',encoding='utf-8'))); dets=[]; empt=0; invalid=0; nonfin=0; top=[]
    for r in rows:
        ds=parse_pred(r.get('prediction_string',''))
        if str(r.get('prediction_string','')).strip() and len(str(r.get('prediction_string','')).strip().split())%5: invalid+=1
        if not ds: empt+=1
        else: top.append(max(d[0] for d in ds))
        for d in ds:
            dets.append(d[0])
            if not math.isfinite(d[0]): nonfin+=1
    return {'rows':len(rows),'detections_total':len(dets),'empty_images':empt,'confidence_sum':float(sum(dets)),'score_median':float(np.median(dets)) if dets else None,'top_score_median':float(np.median(top)) if top else None,'invalid_strings':invalid,'nonfinite':nonfin}

def iou(a,b):
    ax,ay,aw,ah=a[1],a[2],a[3],a[4]; bx,by,bw,bh=b[1],b[2],b[3],b[4]
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix=max(0,min(ax2,bx2)-max(ax,bx)); iy=max(0,min(ay2,by2)-max(ay,by)); inter=ix*iy; union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0

def greedy(ref,cand,thr=0.5):
    pairs=[]; usedr=set(); usedc=set(); allp=[]
    for i,r in enumerate(ref):
        for j,c in enumerate(cand):
            v=iou(r,c)
            if v>=thr: allp.append((v,i,j))
    for v,i,j in sorted(allp, reverse=True):
        if i not in usedr and j not in usedc: usedr.add(i); usedc.add(j); pairs.append((i,j,v))
    return pairs,usedr,usedc

def spatial_compare(ref_map,cand_map,rows):
    rt=ct=mat=0; ious=[]; score_deltas=[]; score_abs=[]; img_mod=0; img_spatial=0; img_conf=0; boxes_mod=0
    for r in rows:
        img=str(r['image_id']); rd=ref_map.get(img,[]); cd=cand_map.get(img,[]); rt+=len(rd); ct+=len(cd)
        pairs,ur,uc=greedy(rd,cd); mat+=len(pairs); spatial_change=(len(rd)-len(ur)+len(cd)-len(uc))>0; conf_change=False
        boxes_mod += (len(rd)-len(ur)+len(cd)-len(uc))
        for i,j,v in pairs:
            ious.append(v); ds=cd[j][0]-rd[i][0]; score_deltas.append(ds); score_abs.append(abs(ds))
            if abs(ds)>1e-4: conf_change=True; boxes_mod+=1
        if spatial_change or conf_change: img_mod+=1
        if spatial_change: img_spatial+=1
        elif conf_change: img_conf+=1
    q=lambda arr,x: float(np.quantile(np.asarray(arr),x)) if arr else None
    return {'ref_detections':rt,'candidate_detections':ct,'matched':mat,'recall_vs_ref':mat/rt if rt else 1,'precision_vs_ref':mat/ct if ct else 1,'dropped_ref_detections':rt-mat,'new_candidate_detections':ct-mat,'matched_iou_median':q(ious,0.5),'matched_score_delta_mean':float(np.mean(score_deltas)) if score_deltas else None,'matched_score_delta_median':q(score_deltas,0.5),'matched_score_delta_p95_abs':q(score_abs,0.95),'images_modified':img_mod,'images_spatial_modified':img_spatial,'images_confidence_only_modified':img_conf,'boxes_modified':boxes_mod}

def cpsr(summary, sp):
    det=summary['detections_total']; emp=summary['empty_images']; conf=summary['confidence_sum']; med=summary['score_median'] or 0; rec=sp['recall_vs_ref']; prec=sp['precision_vs_ref']
    hard=[]
    if det<=100 or emp>=1800 or conf<50 or rec<0.05: hard.append('collapse')
    if det<1600 or emp>950 or conf<620 or rec<0.85: hard.append('suppression')
    if det>2100 or emp<730 or conf>950 or med>0.43: hard.append('inflation')
    if prec<0.85: hard.append('low_precision')
    score=0
    if hard: score=60
    else:
        if not(1750<=det<=1950): score+=8
        if not(760<=emp<=830): score+=8
        if not(700<=conf<=825): score+=8
        if not(0.35<=med<=0.42): score+=5
        if rec<0.97: score+=20
        if prec<0.94: score+=15
    zone='green' if score<=10 and not hard else ('orange' if score<=25 and not hard else 'red')
    return {'cpsr_score':score,'zone':zone,'hard_reject_flags':hard,'green':zone=='green'}

# Main
start=time.time()
pre={'run_id':RUN_ID,'device':DEVICE,'no_submit':True,'forbidden_test_targets':True,'seed':SEED}
write_json(AUDIT_DIR/'preflight_guardrails.json', pre)
max20=find_by_sha(SHA20,'model_final.pth')
ann=find_competition_files(); dataset=build_unlearn_dataset(ann)
write_json(AUDIT_DIR/'preflight_data_contract.json', {'annotations':str(ann),'unlearn_count':len(dataset),'training_data':'unlearn_set_only','test_used_for_training':False,'teacher_targets_on_test':False,'pseudo_labels_test':False,'r02_target':False})
write_json(AUDIT_DIR/'preflight_rules_safety.json', {'no_submit':True,'no_sweep':True,'no_calibration':True,'no_postprocess_extra':True,'official_checkpoint_only':'iter50','diagnostic_only_iters':[10,25]})
write_json(AUDIT_DIR/'input_manifest_sha256.json', {'max20_checkpoint':str(max20),'max20_sha256':sha256_file(max20),'annotation_path':str(ann)})
student=load_model(max20); teacher=load_model(max20); set_scope(student)
sa=scope_audit(student); write_json(AUDIT_DIR/'trainable_scope_audit_before.json', sa)
if not(sa['ok_names'] and sa['ok_total']): raise RuntimeError('bad trainable scope')
official=train_one(student, teacher, dataset)
write_json(AUDIT_DIR/'effect_size_logits_iter50.json', effect_logits_audit(teacher, student, dataset))
# Scope after / diff
base_state=torch.load(max20,map_location='cpu')['model']; final_state=student.state_dict(); diffs=[]; forbidden=[]
for k,v in final_state.items():
    b=base_state[k].to(v.device) if k in base_state else None
    if b is None: continue
    md=float((v.detach()-b).abs().max().cpu())
    if md>0: diffs.append({'name':k,'max_abs_diff':md,'allowed':k in ALLOWED})
    if k not in ALLOWED and md>1e-12: forbidden.append({'name':k,'max_abs_diff':md})
write_json(AUDIT_DIR/'trainable_scope_audit_after_iter50.json', {'changed_tensors':diffs,'forbidden_changes':forbidden,'ok':len(forbidden)==0})
# Inference official iter50 and max20 ref
iter_csv=CSV_DIR/'distill_preserve_iter50_submission.csv'; max20_csv=CSV_DIR/'max20_reference_reinfer_submission.csv'
# save compatible official model_final for predictor
ck=torch.load(official,map_location='cpu'); model_final=CKPT_DIR/'model_final_iter50_official.pth'; torch.save({'model':{k:v.detach().cpu() for k,v in student.state_dict().items()},'iteration':50}, model_final)
run_inference(model_final, iter_csv)
run_inference(max20, max20_csv)
r02_path=find_by_sha(SHA_R02,'*.csv')
# r02 is loaded only here, after training and inference, for final comparison/audit; never as target.
manifest=json.loads((AUDIT_DIR/'input_manifest_sha256.json').read_text())
manifest.update({'r02_csv_final_audit_only':str(r02_path),'r02_sha256':sha256_file(r02_path)})
write_json(AUDIT_DIR/'input_manifest_sha256.json', manifest)
rows=read_sample(); imap=load_map(iter_csv); m20=load_map(max20_csv); r02=load_map(r02_path)
sum_iter=summarize_csv(iter_csv); sum_max=summarize_csv(max20_csv); sum_r02=summarize_csv(r02_path)
sp20=spatial_compare(m20,imap,rows); spr02=spatial_compare(r02,imap,rows)
cps=cpsr(sum_iter,sp20)
efflog=json.loads((AUDIT_DIR/'effect_size_logits_iter50.json').read_text())
effect={'behavior_vs_max20':sp20,'logits_unlearn':efflog,'classification':None}
if not cps['green']: cat='CPSR orange/rouge'
elif sp20['boxes_modified']<=10 and sp20['images_modified']<=10 and efflog['effect_ratio']<1.2: cat='CPSR green mais effet negligeable'
elif efflog['effect_ratio']>=2.0 and (efflog['poison']['mean_signed'] or 0)<0: cat='CPSR green avec effet localise poison'
else: cat='CPSR green avec derive globale'
effect['classification']=cat
final={'run_id':RUN_ID,'runtime_sec':time.time()-start,'summary_iter50':sum_iter,'summary_max20_reinfer':sum_max,'summary_r02':sum_r02,'spatial_vs_max20':sp20,'spatial_vs_r02':spr02,'cpsr':cps,'effect_size_audit':effect,'final_category':cat,'question1_effect_real': sp20['boxes_modified']>10 or sp20['images_modified']>10 or (efflog['poison']['mean_abs'] or 0)>1e-6,'question2_effect_poison_concentrated': efflog['effect_ratio']>=2.0 and (efflog['poison']['mean_signed'] or 0)<0,'submitted':False}
write_json(AUDIT_DIR/'final_audit.json', final)
# report
md=[]
md.append('# distill_preserve_max20_001 audit')
md.append('Mode: no-submit strict. Aucune soumission. iter50 seul checkpoint officiel.')
md.append(f"Final category: {cat}")
md.append('## Scope')
md.append(json.dumps({'before':sa,'after_ok':len(forbidden)==0,'forbidden_changes':forbidden[:5]},indent=2))
md.append('## CPSR')
md.append(json.dumps(cps,indent=2))
md.append('## Iter50 summary')
md.append(json.dumps(sum_iter,indent=2))
md.append('## Spatial vs MAX20')
md.append(json.dumps(sp20,indent=2))
md.append('## Spatial vs r02')
md.append(json.dumps(spr02,indent=2))
md.append('## Effect size audit')
md.append(json.dumps(effect,indent=2))
md.append('## Questions')
md.append(f"Q1 effet reel mesurable: {final['question1_effect_real']}")
md.append(f"Q2 effet principalement poison: {final['question2_effect_poison_concentrated']}")
(REPORT_DIR/'distill_preserve_max20_001_audit.md').write_text('\n\n'.join(md),encoding='utf-8')
# manifest
mans=[]
for p in OUT.rglob('*'):
    if p.is_file(): mans.append({'path':str(p.relative_to(OUT)),'sha256':sha256_file(p),'size':p.stat().st_size})
write_json(OUT/'sha256_manifest.json', {'files':mans})
print(json.dumps({'DONE':True,'category':cat,'cpsr':cps,'effect_ratio':efflog['effect_ratio'],'submitted':False},indent=2))
