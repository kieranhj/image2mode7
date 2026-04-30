@echo off
REM Wrapper around `python -m modal` that forces UTF-8 so Modal's
REM Unicode output (checkmarks, emoji) doesn't crash on the Windows
REM console default codepage. Use exactly like the modal CLI:
REM     modal run modal\prep_gold.py
REM     modal volume get teletext-m1 gold.npz .
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -m modal %*
