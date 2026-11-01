@echo off
REM One-click demo on Windows: creates the venv on first run (CPU build of PyTorch, ~200 MB), then runs the loop on synthetic data.
REM For a CUDA GPU, replace ".../whl/cpu" below with ".../whl/cu128" (about 2.5 GB download).
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
python scripts\demo_loop.py --root demo_runs %*
start "" "%CD%\demo_runs\report"
pause
