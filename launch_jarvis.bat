@echo off
REM JARVIS Launcher - Double-click to start JARVIS
cd /d "%~dp0"

echo ============================================================
echo   JARVIS - Unified Local AI Agent
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or later
    pause
    exit /b 1
)

REM Check if jarvis_agent.py exists
if not exist "jarvis_agent.py" (
    echo ERROR: jarvis_agent.py not found in current directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Starting JARVIS server...
echo.
echo Web interface will be available at: http://127.0.0.1:7860
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

python jarvis_agent.py

if errorlevel 1 (
    echo.
    echo ERROR: JARVIS failed to start
    echo Check the error messages above
    pause
    exit /b 1
)

pause

