@echo off
REM JARVIS Launcher - Double-click to start JARVIS (FastAPI + React)
cd /d "%~dp0"

echo ============================================================
echo   JARVIS - Unified Local AI Agent (FastAPI + React)
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

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if backend/main.py exists
if not exist "backend\main.py" (
    echo ERROR: backend\main.py not found
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo Installing React dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install React dependencies
        pause
        exit /b 1
    )
    cd ..
)

echo Starting JARVIS backend and frontend...
echo.
echo Backend will be available at: http://127.0.0.1:8000
echo Frontend will be available at: http://127.0.0.1:3000
echo Browser will open automatically once everything is ready...
echo.
echo ============================================================
echo.

REM Kill any existing processes on ports 8000 and 3000
echo Stopping any existing JARVIS processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Start backend in minimized window
echo Starting backend...
start "JARVIS Backend" /MIN cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

REM Give backend time to actually start the process and bind to port
echo Waiting for backend process to start...
timeout /t 10 /nobreak >nul

REM Wait for backend to be fully ready (not just started, but models loaded and ready)
echo Waiting for backend to be fully ready...
echo This may take 30-60 seconds while models load...
powershell -Command "$maxAttempts = 30; $attempt = 0; $backendReady = $false; while (-not $backendReady -and $attempt -lt $maxAttempts) { Start-Sleep -Seconds 2; $attempt++; try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3 -ErrorAction Stop; if ($response.status -eq 'healthy' -or $response.ready -eq $true) { $backendReady = $true; Write-Host 'Backend is ready!' } } catch { $ex = $_.Exception; if ($ex.Response) { $statusCode = [int]$ex.Response.StatusCode; if ($statusCode -eq 503) { if ($attempt %% 5 -eq 0) { Write-Host \"Backend is initializing... (attempt $attempt/$maxAttempts)\" } } else { if ($attempt %% 5 -eq 0) { Write-Host \"Waiting for backend... (attempt $attempt/$maxAttempts) - HTTP $statusCode\" } } } else { if ($attempt %% 5 -eq 0) { Write-Host \"Waiting for backend to start... (attempt $attempt/$maxAttempts)\" } } } }; if (-not $backendReady) { Write-Host 'WARNING: Backend did not become ready in time. It may still be loading models. Check minimized windows for errors.' }"

REM Start frontend in minimized window
echo Starting frontend...
cd frontend
start "JARVIS Frontend" /MIN cmd /k "cd /d %~dp0\frontend && npm start"
cd ..

REM Wait for frontend to be ready, then open browser
echo Waiting for frontend to be ready...
powershell -Command "$maxAttempts = 45; $attempt = 0; $frontendReady = $false; while (-not $frontendReady -and $attempt -lt $maxAttempts) { Start-Sleep -Seconds 2; $attempt++; try { $response = Invoke-WebRequest -Uri 'http://localhost:3000' -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop; if ($response.StatusCode -eq 200) { $frontendReady = $true; Write-Host 'Frontend is ready! Opening browser...'; Start-Process 'http://localhost:3000' } } catch {} }; if (-not $frontendReady) { Write-Host 'Frontend did not start in time. Please open http://localhost:3000 manually.' }"

echo.
echo ============================================================
echo   JARVIS is running!
echo   Backend: http://127.0.0.1:8000
echo   Frontend: http://localhost:3000
echo   Browser should open automatically
echo ============================================================
echo.
echo Services are running in minimized windows (check taskbar).
echo To stop JARVIS, click the "Close JARVIS" button in the browser
echo or run: stop_jarvis.bat
echo.
echo This window will close automatically in 3 seconds...
timeout /t 3 >nul
exit
