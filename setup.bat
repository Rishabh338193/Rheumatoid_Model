@echo off
REM Rheumatoid Arthritis Prediction - Setup Script (Windows)
REM This script sets up and runs the complete project

echo ==========================================
echo RA Prediction System - Setup
echo ==========================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

python --version
echo Python found
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed successfully
echo.

REM Generate dataset
echo Generating synthetic RA dataset...
cd data
python generate_dataset.py

if %errorlevel% neq 0 (
    echo Failed to generate dataset
    pause
    exit /b 1
)

cd ..
echo Dataset generated successfully
echo.

REM Train model
echo Training Random Forest model...
cd notebooks
python train_model.py

if %errorlevel% neq 0 (
    echo Failed to train model
    pause
    exit /b 1
)

cd ..
echo Model trained successfully
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To start the application:
echo.
echo 1. Start Backend API:
echo    cd backend
echo    python app.py
echo.
echo 2. Open Frontend (choose one):
echo    Option A: Double-click frontend/index.html
echo    Option B: cd frontend ^&^& streamlit run streamlit_app.py
echo.
echo ==========================================
pause
