@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === bigrun: checking Python ===
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo During install, check "Add python.exe to PATH". Then re-run this script.
    exit /b 1
)
python --version

if not exist "venv\" (
    echo === bigrun: creating venv ===
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create venv.
        exit /b 1
    )
) else (
    echo === bigrun: venv already exists, reusing it ===
)

call venv\Scripts\activate.bat

echo === bigrun: installing dependencies (this may take a minute) ===
python -m pip install --upgrade pip --quiet
python -m pip install --quiet matplotlib "psutil"
python -m pip install --quiet --upgrade --force-reinstall "networkx @ git+https://github.com/yulie0101/networkx@add-gabow-maximum-cardinality-matching"
if errorlevel 1 (
    echo.
    echo Failed to install networkx from the branch. Check your internet
    echo connection and that git is installed (https://git-scm.com/downloads).
    exit /b 1
)

echo === bigrun: running experiments (resumable -- interrupting and re-running this script picks up where it left off) ===
python bigrun.py
if errorlevel 1 (
    echo.
    echo bigrun.py exited with an error -- see above. Re-running this script
    echo will resume from whatever was already completed.
    exit /b 1
)

echo === bigrun: generating plots ===
python make_plots.py

echo.
echo === bigrun: done ===
echo Results: %~dp0results\bigrun_correctness.csv
echo          %~dp0results\bigrun_complexity.csv
echo Plots:   %~dp0results\plots\
echo Environment info: %~dp0results\environment.json

endlocal
