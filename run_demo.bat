@echo off
REM One-click demo on Windows: creates the venv on first run, then runs the two-round loop on synthetic data.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
python scripts\demo_loop.py --root demo_runs
start "" "%CD%\demo_runs\report"
pause
