#!/usr/bin/env bash
# One-command demo on Linux/macOS: creates the venv on first run, then runs the two-round loop on synthetic data.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python scripts/demo_loop.py --root demo_runs "$@"
echo "report: demo_runs/report/report.md"
