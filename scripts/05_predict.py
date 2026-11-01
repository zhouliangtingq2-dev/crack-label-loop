#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict crack masks for a folder of images and write them as editable label PNGs.

Post-processing = "protected narrowing": thick predicted regions are eroded, but the 1-pixel skeleton
of the original prediction is always kept, so thin cracks never disappear or break.
It differs from simple erosion:
  simple erosion:       thin cracks may disappear or break
  protected erosion:    eroded mask OR skeleton(original mask)

Outputs:
  mask_id/       binary 0/1 PNG
  affinity_png/  transparent PNG, crack colour #009BFF -> open in Affinity/GIMP/Photoshop, correct, save back as a label
  overlay/       preview over original image
  probability/   grayscale probability map

Usage:
  python scripts/05_predict.py --checkpoint runs/r1/best_model.pth --images data/new_images --out runs/pred_new --erode-pixels 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

try:
    from skimage.morphology import skeletonize
except Exception as exc:  # pragma: no cover
    skeletonize = None
    _SKIMAGE_ERROR = exc
else:
    _SKIMAGE_ERROR = None

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
CRACK_RGB = (0, 155, 255)  # #009BFF


def list_images(d: Path):
    if not d.exists():
        raise FileNotFoundError(f'Image folder not found: {d}')
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def make_model(model: str, encoder: str):
    name = model.lower()
    if name in ['unet', 'u-net']:
        return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    if name in ['unetplusplus', 'unet++', 'u-net++']:
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    if name in ['deeplabv3plus', 'deeplabv3+', 'deeplab']:
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    raise ValueError(f'Unknown model: {model}')


def pad(img: np.ndarray, patch: int):
    h, w = img.shape[:2]
    ph = (patch - h % patch) % patch
    pw = (patch - w % patch) % patch
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, h, w


def positions(h: int, w: int, patch: int, overlap: int):
    stride = patch - overlap
    if stride <= 0:
        raise ValueError('overlap must be smaller than patch-size')
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)
    return ys, xs


@torch.no_grad()
def pred_one(model, img: np.ndarray, patch: int, overlap: int, batch_size: int, device, amp=True):
    imgp, oh, ow = pad(img, patch)
    h, w = imgp.shape[:2]
    ys, xs = positions(h, w, patch, overlap)
    tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    prob = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    batch, coords = [], []

    def flush():
        nonlocal batch, coords, prob, cnt
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            p = torch.sigmoid(model(x)[:, 0]).float().cpu().numpy()
        for pm, (y, x0) in zip(p, coords):
            prob[y:y + patch, x0:x0 + patch] += pm
            cnt[y:y + patch, x0:x0 + patch] += 1
        batch, coords = [], []

    for y in ys:
        for x0 in xs:
            batch.append(tf(image=imgp[y:y + patch, x0:x0 + patch])['image'])
            coords.append((y, x0))
            if len(batch) >= batch_size:
                flush()
    flush()
    return (prob / np.maximum(cnt, 1e-6))[:oh, :ow]


def remove_small(mask: np.ndarray, min_area: int):
    mask = mask.astype(bool)
    if min_area <= 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = True
    return out


def kernel(kind: str):
    kind = kind.lower()
    if kind == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if kind == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if kind == 'square':
        return np.ones((3, 3), np.uint8)
    raise ValueError('--erode-kernel must be cross, ellipse, or square')


def protected_narrow_mask(
    mask: np.ndarray,
    erode_pixels: int,
    erode_kernel: str,
    skeleton_min_area: int = 0,
    skeleton_dilate: int = 0,
):
    """Shrink thick crack regions but preserve a 1-pixel skeleton from the original mask.

    out = erode(mask, erode_pixels) OR skeleton(mask)

    If erode_pixels=2, thick regions become narrower, but a 1-pixel line remains
    even where simple erosion would delete thin cracks.
    """
    mask = mask.astype(bool)
    if erode_pixels <= 0:
        return mask
    if skeletonize is None:
        raise RuntimeError(
            'scikit-image skeletonize is unavailable. Please run: pip install scikit-image\n'
            f'Import error was: {_SKIMAGE_ERROR}'
        )

    # Shrink all thick regions.
    eroded = cv2.erode(mask.astype(np.uint8), kernel(erode_kernel), iterations=int(erode_pixels)).astype(bool)

    # Preserve the original centerline so thin cracks do not disappear.
    skel = skeletonize(mask).astype(bool)
    if skeleton_min_area > 0:
        skel = remove_small(skel, skeleton_min_area)
    if skeleton_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = cv2.dilate(skel.astype(np.uint8), k, iterations=int(skeleton_dilate)).astype(bool)

    return eroded | skel


def to_rgba(mask: np.ndarray):
    out = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
    out[mask] = (CRACK_RGB[0], CRACK_RGB[1], CRACK_RGB[2], 255)
    return out


def overlay(img: np.ndarray, mask: np.ndarray, alpha=0.85):
    out = img.astype(np.float32).copy()
    color = np.zeros_like(out)
    color[mask] = CRACK_RGB
    out[mask] = (1 - alpha) * out[mask] + alpha * color[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--patch-size', type=int, default=None)
    ap.add_argument('--overlap', type=int, default=96)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--min-area', type=int, default=5)
    ap.add_argument('--erode-pixels', type=int, default=2, help='Protected erosion amount. 2 is recommended for "narrow but keep 1px".')
    ap.add_argument('--erode-kernel', default='cross', choices=['cross', 'ellipse', 'square'], help='cross is least aggressive; square is strongest')
    ap.add_argument('--skeleton-min-area', type=int, default=0, help='Remove tiny skeleton fragments. 0 keeps all skeleton pieces.')
    ap.add_argument('--skeleton-dilate', type=int, default=0, help='0 keeps minimum 1 px. 1 makes protected skeleton about 3 px, usually not needed.')
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu')
    model_name = ck.get('model', 'unet')
    enc = ck.get('encoder', 'resnet34')
    patch = args.patch_size or int(ck.get('patch_size', 256))
    thr = args.threshold if args.threshold is not None else float(ck.get('threshold', 0.35))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(model_name, enc)
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()

    for sub in ['mask_id', 'affinity_png', 'overlay', 'probability']:
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    imgs = list_images(args.images)
    print(
        f'Predict {len(imgs)} images | threshold={thr} | protected_erode_pixels={args.erode_pixels} | '
        f'kernel={args.erode_kernel} | skeleton_dilate={args.skeleton_dilate} | device={device}'
    )
    for p in tqdm(imgs):
        img = np.array(Image.open(p).convert('RGB'))
        pr = pred_one(model, img, patch, args.overlap, args.batch_size, device, not args.no_amp)
        m0 = remove_small(pr >= thr, args.min_area)
        m = protected_narrow_mask(
            m0,
            erode_pixels=args.erode_pixels,
            erode_kernel=args.erode_kernel,
            skeleton_min_area=args.skeleton_min_area,
            skeleton_dilate=args.skeleton_dilate,
        )
        # Remove tiny fragments after union, but keep threshold conservative.
        m = remove_small(m, max(1, args.min_area // 2))
        Image.fromarray(m.astype(np.uint8)).save(args.out / 'mask_id' / f'{p.stem}.png')
        Image.fromarray(to_rgba(m)).save(args.out / 'affinity_png' / f'{p.stem}.png')
        Image.fromarray(overlay(img, m)).save(args.out / 'overlay' / f'{p.stem}.png')
        Image.fromarray(np.clip(pr * 255, 0, 255).astype(np.uint8)).save(args.out / 'probability' / f'{p.stem}.png')
    print('Done:', args.out)


if __name__ == '__main__':
    main()
