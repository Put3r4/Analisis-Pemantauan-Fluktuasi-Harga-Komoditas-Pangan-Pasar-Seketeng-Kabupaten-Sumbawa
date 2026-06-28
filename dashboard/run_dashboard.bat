@echo off
REM PANTAU PASAR Dashboard - Quick Start Script
REM Windows Batch File

echo.
echo ================================================
echo   PANTAU PASAR Dashboard
echo   Sistem Pendukung Keputusan Harga Pangan
echo   Pasar Seketeng, Kabupaten Sumbawa
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo.
    echo [2/4] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo.
echo [4/4] Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check requirements.txt
    pause
    exit /b 1
)

REM Clear screen for clean dashboard launch
cls

echo.
echo ================================================
echo   Starting PANTAU PASAR Dashboard...
echo ================================================
echo.
echo Dashboard will open automatically in your browser
echo Default URL: http://localhost:8501
echo.
echo Press CTRL+C to stop the dashboard
echo.

REM Run Streamlit
streamlit run app.py

REM If Streamlit exits
echo.
echo Dashboard stopped.
pause
