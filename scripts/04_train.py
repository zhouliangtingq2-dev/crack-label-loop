#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a binary crack segmentation model (U-Net / U-Net++ / DeepLabV3+ from segmentation_models_pytorch).

Design points for thin cracks:
  * patches are sampled around crack pixels with probability --pos-prob (default 0.85) so that the
    extremely sparse positive class is seen often enough;
  * loss = 0.6 * BCE(pos_weight) + 1.0 * soft Dice, computed only on valid pixels (255 = ignore);
  * validation uses a sliding grid over whole images at the same patch size;
  * mixed precision on CUDA; checkpoints store model name, encoder, patch size and threshold so that
    05_predict.py needs no extra arguments.

Usage:
  python scripts/04_train.py --data data/processed_aug --splits data/splits_aug --out runs/r1 --model unet --encoder resnet34 --patch-size 512 --batch-size 8 --epochs 80 --steps-per-epoch 300
"""
from __future__ import annotations
import argparse, csv, random, time
from pathlib import Path
from typing import List, Dict, Tuple
import cv2, numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

IMG_EXTS=['.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp']
IGNORE=255

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def find_file(folder:Path, stem:str):
    if Path(stem).suffix.lower() in IMG_EXTS: stem=Path(stem).stem   # split lists may carry an extension
    for ext in IMG_EXTS:
        p=folder/f'{stem}{ext}'
        if p.exists(): return p
    raise FileNotFoundError(f'Cannot find {stem} in {folder}')

def read_split(p:Path)->List[str]:
    return [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def read_pair(data:Path, stem:str):
    img=np.array(Image.open(find_file(data/'images', stem)).convert('RGB'))
    m=np.array(Image.open(find_file(data/'masks_id', stem)).convert('L'))
    if img.shape[:2] != m.shape[:2]: raise ValueError(f'size mismatch {stem}')
    # Values: 0 background, 1 crack, 255 ignore. Any 2 from old masks is also ignored.
    m=np.where(m==1,1,np.where(m==0,0,IGNORE)).astype(np.uint8)
    return img,m

def pad(img,m,patch):
    h,w=m.shape; ph=max(0,patch-h); pw=max(0,patch-w)
    if ph or pw:
        img=cv2.copyMakeBorder(img,0,ph,0,pw,cv2.BORDER_REFLECT_101)
        m=cv2.copyMakeBorder(m,0,ph,0,pw,cv2.BORDER_CONSTANT,value=0)
    return img,m

def build_train_tf():
    """Photometric + mild geometric augmentation; written to work with albumentations 1.4 and 2.x."""
    try:   # albumentations >= 2.0
        noise = A.GaussNoise(std_range=(0.005, 0.02), p=0.15)
        affine = A.Affine(translate_percent=0.02, scale=(0.92, 1.08), rotate=(-8, 8), border_mode=cv2.BORDER_REFLECT_101, fill_mask=0, p=0.45)
    except TypeError:   # albumentations 1.x
        noise = A.GaussNoise(var_limit=(2.0, 12.0), p=0.15)
        affine = A.Affine(translate_percent=0.02, scale=(0.92, 1.08), rotate=(-8, 8), mode=cv2.BORDER_REFLECT_101, mask_value=0, p=0.45)
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.25), A.RandomRotate90(p=0.25), affine,
        A.RandomBrightnessContrast(0.12, 0.12, p=0.35), A.RandomGamma((90, 110), p=0.25), noise,
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

class PatchDS(Dataset):
    def __init__(self,data:Path,stems:List[str],patch:int,steps:int,pos_prob:float=0.85,augment=True):
        self.samples=[]; self.patch=patch; self.steps=steps; self.pos_prob=pos_prob
        for s in tqdm(stems,desc='load train'):
            img,m=read_pair(data,s); img,m=pad(img,m,patch)
            ys,xs=np.where(m==1)
            pos=np.stack([ys,xs],axis=1) if len(ys) else np.zeros((0,2),dtype=np.int32)
            self.samples.append((s,img,m,pos))
        self.tf=build_train_tf() if augment else A.Compose([A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
    def __len__(self): return self.steps
    def __getitem__(self,idx):
        s,img,m,pos=self.samples[random.randrange(len(self.samples))]
        h,w=m.shape; p=self.patch
        if len(pos)>0 and random.random()<self.pos_prob:
            cy,cx=pos[random.randrange(len(pos))]
            y0=int(cy-random.randint(p//4,3*p//4)); x0=int(cx-random.randint(p//4,3*p//4))
        else:
            y0=random.randint(0,max(0,h-p)); x0=random.randint(0,max(0,w-p))
        y0=max(0,min(y0,h-p)); x0=max(0,min(x0,w-p))
        ci=img[y0:y0+p,x0:x0+p]; cm=m[y0:y0+p,x0:x0+p]
        out=self.tf(image=ci, mask=cm)
        return out['image'], out['mask'].long()

class GridDS(Dataset):
    def __init__(self,data:Path,stems:List[str],patch:int,overlap:int=128):
        self.items=[]; self.images={}; self.patch=patch
        stride=patch-overlap
        for s in tqdm(stems,desc='load val'):
            img,m=read_pair(data,s); img,m=pad(img,m,patch); h,w=m.shape
            ys=list(range(0,max(1,h-patch+1),stride)); xs=list(range(0,max(1,w-patch+1),stride))
            if ys[-1]!=h-patch: ys.append(h-patch)
            if xs[-1]!=w-patch: xs.append(w-patch)
            self.images[s]=(img,m)
            for y in ys:
                for x in xs: self.items.append((s,y,x))
        self.tf=A.Compose([A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        s,y,x=self.items[idx]; img,m=self.images[s]; p=self.patch
        out=self.tf(image=img[y:y+p,x:x+p], mask=m[y:y+p,x:x+p])
        return out['image'], out['mask'].long()

def make_model(model,encoder,pretrained=True):
    w='imagenet' if pretrained else None
    if model.lower() in ['unet','u-net']:
        return smp.Unet(encoder_name=encoder, encoder_weights=w, in_channels=3, classes=1)
    if model.lower() in ['unetplusplus','unet++','u-net++']:
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=w, in_channels=3, classes=1)
    if model.lower() in ['deeplabv3plus', 'deeplabv3+', 'deeplab']:
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=w, in_channels=3, classes=1)
    raise ValueError(model)

class Loss(nn.Module):
    def __init__(self,pos_weight=8.0,bce_weight=0.6,dice_weight=1.0):
        super().__init__(); self.pos_weight=pos_weight; self.bw=bce_weight; self.dw=dice_weight
    def forward(self,logits,target):
        logits=logits[:,0]
        valid=target!=IGNORE
        if valid.sum()==0: return logits.sum()*0
        y=(target==1).float()
        pw=torch.tensor(self.pos_weight,device=logits.device,dtype=logits.dtype)
        bce=F.binary_cross_entropy_with_logits(logits[valid], y[valid], pos_weight=pw)
        prob=torch.sigmoid(logits)
        pv=prob[valid]; yv=y[valid]
        inter=(pv*yv).sum(); denom=pv.sum()+yv.sum()
        dice=1-(2*inter+1)/(denom+1)
        return self.bw*bce+self.dw*dice

def metrics(logits,target,thr=0.35):
    prob=torch.sigmoid(logits[:,0]); valid=target!=IGNORE; y=target==1; pred=prob>thr
    pred=pred[valid]; y=y[valid]
    tp=(pred & y).sum().double(); fp=(pred & ~y).sum().double(); fn=(~pred & y).sum().double()
    iou=(tp/(tp+fp+fn+1e-7)).item(); dice=(2*tp/(2*tp+fp+fn+1e-7)).item(); rec=(tp/(tp+fn+1e-7)).item(); prec=(tp/(tp+fp+1e-7)).item()
    return iou,dice,rec,prec

def run_epoch(model,loader,opt,scaler,loss_fn,device,train=True,amp=True,thr=0.35):
    model.train(train); losses=[]; sums=np.zeros(4); n=0
    pbar=tqdm(loader,desc='train' if train else 'val',leave=False)
    for img,m in pbar:
        img=img.to(device); m=m.to(device)
        if train: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp and device.type=='cuda'):
                logits=model(img); loss=loss_fn(logits,m)
            if train:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            vals=metrics(logits.detach(),m,thr); sums+=np.array(vals); n+=1
        pbar.set_postfix(loss=np.mean(losses),iou=sums[0]/n,dice=sums[1]/n)
    return float(np.mean(losses)), tuple((sums/max(1,n)).tolist())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,type=Path); ap.add_argument('--splits',required=True,type=Path); ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--model',default='unet'); ap.add_argument('--encoder',default='resnet34')
    ap.add_argument('--patch-size',type=int,default=512); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--steps-per-epoch',type=int,default=400)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--num-workers',type=int,default=2); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--pos-weight',type=float,default=8.0); ap.add_argument('--pos-prob',type=float,default=0.85,help='probability that a training patch is centred near a crack pixel')
    ap.add_argument('--threshold',type=float,default=0.35); ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--no-pretrained',action='store_true',help='do not download ImageNet encoder weights (CI / offline)')
    args=ap.parse_args(); set_seed(args.seed); args.out.mkdir(parents=True,exist_ok=True)
    train=read_split(args.splits/'train.txt'); val=read_split(args.splits/'val.txt')
    print('train',len(train),'val',len(val),'data',args.data)
    tr=PatchDS(args.data,train,args.patch_size,args.steps_per_epoch*args.batch_size,pos_prob=args.pos_prob,augment=True)
    va=GridDS(args.data,val,args.patch_size,overlap=min(128,args.patch_size//2))
    tr_loader=DataLoader(tr,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    va_loader=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(args.model,args.encoder,pretrained=not args.no_pretrained).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    loss_fn=Loss(pos_weight=args.pos_weight)
    use_scaler=(not args.no_amp and device.type=='cuda')
    try: scaler=torch.amp.GradScaler('cuda',enabled=use_scaler)
    except (AttributeError,TypeError): scaler=torch.cuda.amp.GradScaler(enabled=use_scaler)
    best=-1; hist=[]
    for ep in range(1,args.epochs+1):
        t0=time.time(); tl,tm=run_epoch(model,tr_loader,opt,scaler,loss_fn,device,True,not args.no_amp,args.threshold)
        vl,vm=run_epoch(model,va_loader,None,scaler,loss_fn,device,False,not args.no_amp,args.threshold); sched.step()
        score=vm[1]
        row={'epoch':ep,'train_loss':tl,'val_loss':vl,'train_iou':tm[0],'train_dice':tm[1],'train_recall':tm[2],'train_precision':tm[3],'val_iou':vm[0],'val_dice':vm[1],'val_recall':vm[2],'val_precision':vm[3],'lr':sched.get_last_lr()[0],'sec':time.time()-t0}
        hist.append(row)
        print(f"Epoch {ep:03d}: train_loss={tl:.4f} val_loss={vl:.4f} val_iou={vm[0]:.4f} val_dice={vm[1]:.4f} recall={vm[2]:.3f} precision={vm[3]:.3f}")
        ck={'model_state':model.state_dict(),'model':args.model,'encoder':args.encoder,'patch_size':args.patch_size,'threshold':args.threshold,'binary_crack':True,'pretrained':not args.no_pretrained}
        torch.save(ck,args.out/'last_model.pth')
        if score>best:
            best=score; torch.save(ck,args.out/'best_model.pth'); print('  saved best')
        with open(args.out/'history.csv','w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
    print('Done. Best val dice:',best)

if __name__=='__main__': main()
