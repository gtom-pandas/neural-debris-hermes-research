# controlled_unlearn_beta005_preserve_001
# Strict no-submit. Training data: unlearn_set only. Student/teacher/init/reference/fallback = beta0p05. No Kaggle submission.
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

RUN_ID='controlled_unlearn_beta005_preserve_001'
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
SHABETA='eff02bca7db558ba1fa31da6a46406009f77dde4be2c5c91b4a6451512762320'
SHA_R02='5ee41b823fdd052fdcf237d98d58dca1ccec9edbf1a82184e9b64934c42dbd60'
ALLOWED=['head.cls_subnet.6.weight','head.cls_subnet.6.bias','head.cls_score.weight','head.cls_score.bias']
EXPECTED_TRAINABLE=606215
SEED=20260620
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

BASELINE={
 'interp20_30_beta0p05': {'detections_total':1831,'empty_images':806,'confidence_sum':750.415,'public':257.6416},
 'r02_cluster_visible_no_power': {'detections_total':1815,'empty_images':791,'confidence_sum':760.942,'public':259.2333},
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

def train_one(student, teacher, dataset, variant):
    # Controlled poison-targeted unlearning from beta0p05 with strong preservation.
    set_scope(student); student.eval(); teacher.eval()
    for p in teacher.parameters(): p.requires_grad=False
    lr=float(variant['lr']); max_iter=int(variant['iters'])
    opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    init={n:p.detach().clone() for n,p in student.named_parameters() if n in ALLOWED}
    logs=[]
    vdir=CKPT_DIR/variant['name']; vdir.mkdir(parents=True, exist_ok=True)
    for it in range(1,max_iter+1):
        item=dataset[(it-1)%len(dataset)]
        img=tensors_from_image(item['path'])
        with torch.no_grad():
            t_feats,t_logits=get_features_and_logits(teacher,img)
        s_feats,s_logits=get_features_and_logits(student,img)
        loss_negative=torch.tensor(0.,device=DEVICE); loss_pres=torch.tensor(0.,device=DEVICE); loss_rank=torch.tensor(0.,device=DEVICE); loss_corr=torch.tensor(0.,device=DEVICE)
        un_ct=pres_ct=rank_ct=0
        for sf,tf,sl,tl in zip(s_feats,t_feats,s_logits,t_logits):
            B,C,H,W=sl.shape
            pm=masks_for_shape(item['boxes_xyxy'],H,W,C,DEVICE)
            out=~pm
            if pm.any():
                # poison zone: push target confidence down, but with low lr/l2 so it cannot globally collapse.
                loss_negative = loss_negative + F.binary_cross_entropy_with_logits(sl[pm], torch.zeros_like(sl[pm])); un_ct+=int(pm.sum())
            with torch.no_grad():
                prob=torch.sigmoid(tl)
                preserve = out & (prob>=0.003)
                if preserve.sum()==0:
                    flat=(prob*out.float()).flatten(); k=min(300,flat.numel()); idx=torch.topk(flat,k).indices; preserve=torch.zeros_like(out.flatten()); preserve[idx]=True; preserve=preserve.view_as(out)
            loss_pres = loss_pres + F.mse_loss(sl[preserve], tl[preserve]); pres_ct+=int(preserve.sum())
            # rank/corridor preserve on high teacher scores outside poison
            with torch.no_grad():
                flat=(torch.sigmoid(tl)*out.float()).flatten(); k=min(512, flat.numel())
                idx=torch.topk(flat,k).indices
                t_top=tl.flatten()[idx]
            s_top=sl.flatten()[idx]
            if s_top.numel()>1:
                loss_rank = loss_rank + F.mse_loss(s_top - s_top.mean(), t_top - t_top.mean()); rank_ct+=int(s_top.numel())
                ratio=torch.sigmoid(s_top)/(torch.sigmoid(t_top)+1e-6)
                loss_corr = loss_corr + F.relu(0.96-ratio).pow(2).mean() + F.relu(ratio-1.01).pow(2).mean()
        l2=torch.tensor(0.,device=DEVICE)
        for n,p in student.named_parameters():
            if n in ALLOWED: l2=l2+F.mse_loss(p,init[n])
        total=1.0*loss_negative + 8.0*loss_pres + 30.0*l2 + 2.0*loss_rank + 3.0*loss_corr
        opt.zero_grad(); total.backward(); grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.0)); opt.step()
        rec={'variant':variant['name'],'iter':it,'lr':lr,'image':Path(item['path']).name,'loss_total':float(total.detach()),'loss_negative_poison':float(loss_negative.detach()),'loss_preserve_outside':float(loss_pres.detach()),'loss_l2sp':float(l2.detach()),'loss_rank_preserve':float(loss_rank.detach()),'loss_corridor':float(loss_corr.detach()),'grad_norm':grad_norm,'poison_logit_count':un_ct,'preserve_logit_count':pres_ct,'rank_count':rank_ct}
        logs.append(rec)
        with open(LOG_DIR/f"train_log_{variant['name']}.jsonl",'a') as f: f.write(json.dumps(rec)+'\n')
    ck={'model':student.state_dict(),'iteration':max_iter,'diagnostic_only':False,'eligible_for_promotion':True,'official_checkpoint':True,'variant':variant,'protocol':RUN_ID}
    out=vdir/'model_final.pth'
    torch.save(ck, out)
    write_json(LOG_DIR/f"train_metrics_summary_{variant['name']}.json", {'variant':variant,'records':logs})
    return out

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
pre={'run_id':RUN_ID,'device':DEVICE,'no_submit':True,'auto_submit':False,'submission_authorized':False,'paid_gpu_authorized':False,'active_run_expected':None,'forbidden_submit_api':True,'seed':SEED}
write_json(AUDIT_DIR/'preflight_guardrails.json', pre)
beta=find_by_sha(SHABETA,'*.pth')
ann=find_competition_files(); dataset=build_unlearn_dataset(ann)
write_json(AUDIT_DIR/'preflight_data_contract.json', {'annotations':str(ann),'unlearn_count':len(dataset),'training_data':'unlearn_set_only','test_used_for_training':False,'teacher_targets_on_test':False,'pseudo_labels_test':False,'r02_target':False,'beta0p05_as_init_teacher_anchor_fallback':True})
write_json(AUDIT_DIR/'preflight_rules_safety.json', {'no_submit':True,'no_parallel_family':True,'no_beta_grid':True,'no_calibration':True,'variants_predeclared':['B1_lr1e-6_iter50','B1_lr1e-6_iter100','B1_lr2e-6_iter50','B1_lr2e-6_iter100']})
write_json(AUDIT_DIR/'input_manifest_sha256.json', {'beta0p05_checkpoint':str(beta),'beta0p05_sha256':sha256_file(beta),'annotation_path':str(ann)})
# Reference inference once for CPSR/fallback, using beta0p05 checkpoint
beta_csv=CSV_DIR/'beta0p05_reference_reinfer_submission.csv'
run_inference(beta, beta_csv)
rows=read_sample(); beta_map=load_map(beta_csv); sum_beta=summarize_csv(beta_csv)
variants=[
 {'name':'B1_lr1e-6_iter50','lr':1e-6,'iters':50},
 {'name':'B1_lr1e-6_iter100','lr':1e-6,'iters':100},
 {'name':'B1_lr2e-6_iter50','lr':2e-6,'iters':50},
 {'name':'B1_lr2e-6_iter100','lr':2e-6,'iters':100},
]
variant_records=[]
for variant in variants:
    # Fresh student from beta for each variant; teacher frozen beta.
    student=load_model(beta); teacher=load_model(beta); set_scope(student)
    sa=scope_audit(student); write_json(AUDIT_DIR/f"{variant['name']}_trainable_scope_before.json", sa)
    if not(sa['ok_names'] and sa['ok_total']): raise RuntimeError('bad trainable scope')
    ckpt=train_one(student, teacher, dataset, variant)
    eff=effect_logits_audit(teacher, student, dataset)
    write_json(AUDIT_DIR/f"{variant['name']}_effect_size_logits.json", eff)
    # Diff/scope audit
    base_state=torch.load(beta,map_location='cpu')['model']; final_state=student.state_dict(); diffs=[]; forbidden=[]
    for k,v in final_state.items():
        b=base_state[k].to(v.device) if k in base_state else None
        if b is None: continue
        md=float((v.detach()-b).abs().max().cpu())
        if md>0: diffs.append({'name':k,'max_abs_diff':md,'allowed':k in ALLOWED})
        if k not in ALLOWED and md>1e-12: forbidden.append({'name':k,'max_abs_diff':md})
    write_json(AUDIT_DIR/f"{variant['name']}_trainable_scope_after.json", {'changed_tensors':diffs,'forbidden_changes':forbidden,'ok':len(forbidden)==0})
    # compatible checkpoint already has model key
    out_csv=CSV_DIR/f"{variant['name']}_submission.csv"
    run_inference(ckpt, out_csv)
    cand_map=load_map(out_csv); summ=summarize_csv(out_csv); sp=spatial_compare(beta_map,cand_map,rows); c=cpsr(summ,sp)
    hard_submit_review = (c['zone'] in ['green','orange'] and summ['invalid_strings']==0 and summ['nonfinite']==0 and sp['recall_vs_ref']>=0.970 and sp['precision_vs_ref']>=0.975 and 1700<=summ['detections_total']<=1900 and 700<=summ['confidence_sum']<=770 and eff['effect_ratio']>=1.5)
    rec={'variant':variant,'checkpoint':str(ckpt),'checkpoint_sha256':sha256_file(ckpt),'csv':str(out_csv),'csv_sha256':sha256_file(out_csv),'summary':summ,'spatial_vs_beta0p05':sp,'cpsr':c,'effect_size_logits':eff,'forbidden_scope_changes':forbidden,'submit_review_eligible':bool(hard_submit_review),'submitted':False}
    variant_records.append(rec); write_json(AUDIT_DIR/f"{variant['name']}_full_audit.json", rec)
# Ranking: prioritize credible departure from plateau while staying in corridor.
def rank_key(r):
    c=r['cpsr']; sp=r['spatial_vs_beta0p05']; eff=r['effect_size_logits']; summ=r['summary']
    penalty=0
    if c['zone']=='red': penalty+=1000
    elif c['zone']=='orange': penalty+=100
    penalty += max(0,0.970-sp['recall_vs_ref'])*1000 + max(0,0.975-sp['precision_vs_ref'])*1000
    # prefer real poison-localized effect, not pure identity; prefer confidence slight decrease not collapse
    mechanism_bonus=min(eff.get('effect_ratio',0),5)*10
    target_conf=735.0
    return penalty + abs(summ['confidence_sum']-target_conf)/10 - mechanism_bonus
ranking=sorted(variant_records, key=rank_key)
for i,r in enumerate(ranking,1): r['rank']=i
eligible=[r for r in ranking if r['submit_review_eligible']]
top=eligible[0] if eligible else None
final={'run_id':RUN_ID,'runtime_sec':time.time()-start,'beta0p05_reference':{'checkpoint':str(beta),'checkpoint_sha256':sha256_file(beta),'csv_reinfer':str(beta_csv),'csv_reinfer_sha256':sha256_file(beta_csv),'summary':sum_beta,'public_score':257.6416},'variants':variant_records,'ranking':[{'rank':r['rank'],'variant':r['variant']['name'],'csv_sha256':r['csv_sha256'],'cpsr_zone':r['cpsr']['zone'],'detections':r['summary']['detections_total'],'empty':r['summary']['empty_images'],'confidence_sum':r['summary']['confidence_sum'],'recall':r['spatial_vs_beta0p05']['recall_vs_ref'],'precision':r['spatial_vs_beta0p05']['precision_vs_ref'],'effect_ratio':r['effect_size_logits']['effect_ratio'],'submit_review_eligible':r['submit_review_eligible']} for r in ranking],'top_submit_review':top,'submitted':False}
write_json(AUDIT_DIR/'final_audit.json', final)
write_json(OUT/'candidate_ranking.json', final['ranking'])
if top:
    write_json(OUT/'top_submit_review.json', {'decision':'prepare_submit_review_wait_human_validation','candidate':top})
else:
    write_json(OUT/'top_submit_review.json', {'decision':'no_submit_candidate','reason':'no variant passed corridor+mechanism gates'})
# CSV ranking
with open(OUT/'candidate_ranking.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(final['ranking'][0].keys()) if final['ranking'] else ['rank'])
    w.writeheader(); w.writerows(final['ranking'])
md=[]
md.append('# controlled_unlearn_beta005_preserve_001 no-submit report')
md.append('Mode: no-submit strict. Aucune soumission Kaggle. Aucune autre famille en parallèle. beta0p05 = init/teacher/ancre/CPSR/fallback.')
md.append(f"Runtime seconds: {final['runtime_sec']:.2f}")
md.append('## Beta0p05 reference')
md.append(json.dumps(final['beta0p05_reference'], indent=2))
md.append('## Candidate ranking')
md.append(json.dumps(final['ranking'], indent=2))
md.append('## Top submit-review')
md.append(json.dumps({'exists': top is not None, 'candidate': top['variant']['name'] if top else None, 'csv': top['csv'] if top else None, 'sha256': top['csv_sha256'] if top else None}, indent=2))
md.append('## Full variant audits')
for r in ranking:
    md.append('### '+r['variant']['name'])
    md.append(json.dumps({'summary':r['summary'],'cpsr':r['cpsr'],'spatial_vs_beta0p05':r['spatial_vs_beta0p05'],'effect_size_logits':r['effect_size_logits'],'csv':r['csv'],'csv_sha256':r['csv_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],'submit_review_eligible':r['submit_review_eligible']}, indent=2))
md.append('## Strategic answers')
if top:
    md.append('Effet poison-targeted/corridor: au moins une variante passe les gates mécaniques; préparer submit-review mais attendre validation humaine exacte.')
else:
    md.append('Aucun candidat submit-review automatique: interpréter la famille selon les métriques ci-dessus avant abandon/révision.')
(REPORT_DIR/'controlled_unlearn_beta005_preserve_001_no_submit_report.md').write_text('\n\n'.join(md),encoding='utf-8')
# manifest
mans=[]
for p in OUT.rglob('*'):
    if p.is_file(): mans.append({'path':str(p.relative_to(OUT)),'sha256':sha256_file(p),'size':p.stat().st_size})
write_json(OUT/'sha256_manifest.json', {'files':mans,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
print(json.dumps({'DONE':True,'run_id':RUN_ID,'submitted':False,'top_submit_review': top['variant']['name'] if top else None,'ranking':final['ranking']},indent=2))
