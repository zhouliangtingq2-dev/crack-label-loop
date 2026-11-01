# -*- coding: utf-8 -*-
"""Small unit tests for the parts of the pipeline that the demo does not exercise by default.
Run:  pytest -q
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_protected_narrow_keeps_thin_lines_and_narrows_thick_ones():
    pred = load("05_predict.py")
    m = np.zeros((64, 64), bool)
    m[10, 5:60] = True                      # 1-px line
    m[30:37, 5:60] = True                   # 7-px thick band
    out = pred.protected_narrow_mask(m, erode_pixels=2, erode_kernel="cross")
    assert out[10, 5:60].sum() >= 50, "thin line must survive protected narrowing"
    thick_rows = out[28:40, 32].sum()
    assert 1 <= thick_rows < 7, f"thick band should be narrowed but not deleted (got {thick_rows} rows)"
    plain = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    assert plain[10].sum() == 0, "sanity: plain erosion deletes the thin line"


def test_make_masks_from_transparent_and_overlay_labels():
    prep = load("01_prepare_dataset.py")
    h, w = 20, 30
    # transparent label: blue crack, yellow spalling
    lab = np.zeros((h, w, 4), np.uint8)
    lab[5, :, :] = (0, 155, 255, 255)
    lab[12:15, 10:20, :] = (255, 255, 0, 255)
    mask, crack, spall = prep.make_masks(lab, 45, 45, False, 0.5, 0.2, 10)
    assert (mask == 1).sum() == w and (mask == 255).sum() == 30 and (mask == 0).sum() == h * w - w - 30
    # the same colours painted over an opaque grey photo
    lab2 = np.full((h, w, 4), 128, np.uint8); lab2[..., 3] = 255
    lab2[5, :, :3] = (0, 155, 255)
    mask2, _, _ = prep.make_masks(lab2, 45, 45, False, 0.5, 0.2, 10)
    assert (mask2 == 1).sum() == w and (mask2 == 255).sum() == 0


def test_split_with_zero_test_ratio(tmp_path):
    data = tmp_path / "processed" / "masks_id"
    data.mkdir(parents=True)
    for i in range(5):
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(data / f"img_{i}.png")
    out = tmp_path / "splits"
    subprocess.run([sys.executable, str(SCRIPTS / "02_make_splits.py"), "--data", str(tmp_path / "processed"), "--out", str(out),
                    "--train", "0.8", "--val", "0.2", "--test", "0"], check=True)
    n = {k: len((out / f"{k}.txt").read_text().split()) for k in ("train", "val", "test")}
    assert n == {"train": 4, "val": 1, "test": 0}
