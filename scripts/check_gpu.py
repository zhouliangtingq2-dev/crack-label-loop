#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys

try:
    import torch
except Exception as e:
    print("PyTorch import failed:", repr(e))
    print("Install PyTorch first (see README).")
    sys.exit(1)

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print("CUDA device count:", n)
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        print(f"[{i}] {prop.name}")
        print(f"    total_memory: {prop.total_memory / 1024**3:.2f} GB")
        print(f"    capability: {prop.major}.{prop.minor}")
        if prop.total_memory >= 11 * 1024**3:
            print("    recommended: patch_size=512, batch_size=8")
        elif prop.total_memory >= 7 * 1024**3:
            print("    recommended: patch_size=512, batch_size=4")
        else:
            print("    recommended: patch_size=384, batch_size=2")

print("\n---- nvidia-smi ----")
try:
    out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    print(out)
except Exception as e:
    print("nvidia-smi failed:", repr(e))
