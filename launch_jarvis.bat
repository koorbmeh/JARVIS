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
echo Browser will open automatically once server is ready...
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

REM Start PowerShell script in background to check for server readiness and open browser
REM This checks every 2 seconds for up to 60 seconds (30 attempts)
start /B powershell -Command "$maxAttempts = 30; $attempt = 0; $ready = $false; while (-not $ready -and $attempt -lt $maxAttempts) { Start-Sleep -Seconds 2; $attempt++; try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:7860' -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop; $ready = $true; Start-Process 'http://127.0.0.1:7860' } catch {} }; if (-not $ready) { Write-Host 'Server did not start in time. Please open http://127.0.0.1:7860 manually.' }"

REM Start Python script (this will show output in this window)
python jarvis_agent.py

if errorlevel 1 (
    echo.
    echo ERROR: JARVIS failed to start
    echo Check the error messages above
    pause
    exit /b 1
)

pause

