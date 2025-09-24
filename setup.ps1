# -----------------------------
# Project Setup Script (Windows PowerShell)
# This script will:
# 1) Create & activate a virtual environment
# 2) Upgrade pip and install requirements
# 3) Run Django migrations
# 4) Train the ML model
# 5) Start the Django development server
#
# Usage: Run this script from the project root
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# -----------------------------

param(
    [switch]$SkipTrain,
    [switch]$SkipRunServer
)

Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path .\venv)) {
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host "[2/5] Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[3/5] Running Django migrations..." -ForegroundColor Cyan
python manage.py makemigrations
python manage.py migrate

if (-not $SkipTrain) {
    Write-Host "[4/5] Training ML model..." -ForegroundColor Cyan
    python train_model.py
} else {
    Write-Host "[4/5] Skipping training as requested." -ForegroundColor Yellow
}

if (-not $SkipRunServer) {
    Write-Host "[5/5] Starting Django development server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    python manage.py runserver
} else {
    Write-Host "[5/5] Skipping server run as requested." -ForegroundColor Yellow
}


