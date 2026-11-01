#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert hand-painted label images into training masks (crack-only).

Input:
  --images : original photos
  --labels : label PNGs painted in any image editor (Affinity Photo, GIMP, Photoshop ...):
             transparent/black background, crack = #009BFF (0,155,255), spalling = #FFFF00 (255,255,0).
             A label may also be the photo with the colours painted on top; the colours are extracted by
             colour distance (and optionally HSV range).

Output:
  out/images        copied original images as PNG
  out/masks_id      0=background, 1=crack, 255=ignore/spalling
  out/preview_color black background + blue crack + yellow ignored spalling
  out/preview_overlay overlay on original image
  out/metadata.csv  pixel counts

Why crack-only?
  Crack pixels are very thin and spalling pixels are rare; a 3-class model easily collapses into the wrong class.
  Spalling is therefore written as 255 (ignore) so that it neither trains nor penalises the crack model.

Usage:
  python scripts/01_prepare_dataset.py --images data/raw_images --labels data/raw_labels --out data/processed
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image

IMG_EXTS = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
CRACK = np.array([0,155,255], dtype=np.int16)
SPALL = np.array([255,255,0], dtype=np.int16)

def list_by_stem(folder: Path) -> Dict[str, Path]:
    d={}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            d[p.stem]=p
    return d

def dist_mask(rgb: np.ndarray, target: np.ndarray, tol: int) -> np.ndarray:
    diff = rgb.astype(np.int16) - target.reshape(1,1,3)
    dist2 = np.sum(diff.astype(np.int32)**2, axis=2)
    return dist2 <= tol*tol

def hsv_like(rgb: np.ndarray):
    a = rgb.astype(np.float32)/255.0
    r,g,b = a[...,0],a[...,1],a[...,2]
    mx=np.max(a,axis=2); mn=np.min(a,axis=2); df=mx-mn
    hue=np.zeros_like(mx)
    m=df>1e-6
    rmax=m&(mx==r); gmax=m&(mx==g); bmax=m&(mx==b)
    hue[rmax]=(60*((g[rmax]-b[rmax])/df[rmax])+360)%360
    hue[gmax]=60*((b[gmax]-r[gmax])/df[gmax])+120
    hue[bmax]=60*((r[bmax]-g[bmax])/df[bmax])+240
    sat=np.zeros_like(mx); sat[mx>1e-6]=df[mx>1e-6]/mx[mx>1e-6]
    val=mx
    return hue,sat,val

def make_masks(label_rgba: np.ndarray, crack_tol:int, spall_tol:int, use_hsv:bool, min_sat:float, min_val:float, alpha_thr:int):
    rgb=label_rgba[...,:3]
    alpha=label_rgba[...,3]
    # Pure black/transparent background is ignored by color matching anyway; alpha protects transparent export.
    valid = alpha > alpha_thr
    if np.mean(alpha <= alpha_thr) < 0.05:
        valid = np.ones(alpha.shape, dtype=bool)
    crack = dist_mask(rgb, CRACK, crack_tol) & valid
    spall = dist_mask(rgb, SPALL, spall_tol) & valid
    if use_hsv:
        h,s,v = hsv_like(rgb)
        vivid = valid & (s >= min_sat) & (v >= min_val)
        crack |= vivid & (h >= 175) & (h <= 225)   # cyan-blue
        spall |= vivid & (h >= 42) & (h <= 72)     # yellow
    # If anti-aliased pixels overlap, yellow/spalling becomes ignore, not crack.
    crack &= ~spall
    mask=np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[crack]=1
    mask[spall]=255
    return mask, crack, spall

def save_preview(mask: np.ndarray, out_path: Path):
    out=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    out[mask==1]=(0,155,255)
    out[mask==255]=(255,255,0)
    Image.fromarray(out).save(out_path)

def save_overlay(img: np.ndarray, mask: np.ndarray, out_path: Path, alpha=0.85):
    out=img.astype(np.float32).copy()
    color=np.zeros_like(out)
    color[mask==1]=(0,155,255)
    color[mask==255]=(255,255,0)
    m=mask>0
    out[m]=(1-alpha)*out[m]+alpha*color[m]
    Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--labels', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--crack-tolerance', type=int, default=45)
    ap.add_argument('--spall-tolerance', type=int, default=45)
    ap.add_argument('--use-hsv', action='store_true')
    ap.add_argument('--min-sat', type=float, default=0.50)
    ap.add_argument('--min-val', type=float, default=0.20)
    ap.add_argument('--alpha-threshold', type=int, default=10)
    ap.add_argument('--overwrite', action='store_true')
    args=ap.parse_args()
    for sub in ['images','masks_id','preview_color','preview_overlay']:
        (args.out/sub).mkdir(parents=True, exist_ok=True)
    images=list_by_stem(args.images); labels=list_by_stem(args.labels)
    common=sorted(set(images)&set(labels))
    if not common:
        raise SystemExit('No matching image/label stems found. Check filenames.')
    rows=[]; total_crack=total_ignore=total_bg=0
    for stem in common:
        img=Image.open(images[stem]).convert('RGB')
        lab=Image.open(labels[stem]).convert('RGBA')
        if img.size != lab.size:
            raise ValueError(f'Size mismatch {stem}: image {img.size}, label {lab.size}')
        img_np=np.array(img); lab_np=np.array(lab)
        mask, crack, spall = make_masks(lab_np, args.crack_tolerance, args.spall_tolerance, args.use_hsv, args.min_sat, args.min_val, args.alpha_threshold)
        out_name=f'{stem}.png'
        img.save(args.out/'images'/out_name)
        Image.fromarray(mask, mode='L').save(args.out/'masks_id'/out_name)
        save_preview(mask, args.out/'preview_color'/out_name)
        save_overlay(img_np, mask, args.out/'preview_overlay'/out_name)
        bg=int(np.sum(mask==0)); cr=int(np.sum(mask==1)); ign=int(np.sum(mask==255))
        total_bg+=bg; total_crack+=cr; total_ignore+=ign
        warn=[]
        if cr==0: warn.append('NO_CRACK')
        if cr/mask.size < 0.001: warn.append('VERY_FEW_CRACK_PIXELS')
        rows.append({'stem':stem,'image_file':images[stem].name,'label_file':labels[stem].name,'width':img.size[0],'height':img.size[1],
                     'background_px':bg,'crack_px':cr,'ignore_spalling_px':ign,'crack_percent':cr/mask.size,'ignore_percent':ign/mask.size,'warning':';'.join(warn)})
    with open(args.out/'metadata.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.out/'missing_files.txt','w',encoding='utf-8') as f:
        if set(images)-set(labels):
            f.write('Images without labels:\n'+'\n'.join(sorted(set(images)-set(labels)))+'\n')
        if set(labels)-set(images):
            f.write('Labels without images:\n'+'\n'.join(sorted(set(labels)-set(images)))+'\n')
    print('Done crack-only dataset.')
    print('Matched pairs:', len(common))
    print('Total background px:', total_bg)
    print('Total crack px:     ', total_crack)
    print('Total ignore/spall px:', total_ignore)
    print('Check:', args.out/'preview_color')
    print('Check:', args.out/'preview_overlay')
    if total_crack==0:
        print('[SEVERE] No crack pixels recognized. Try --use-hsv or raise --crack-tolerance.')

if __name__=='__main__':
    main()
