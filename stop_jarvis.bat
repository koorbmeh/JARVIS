@echo off
REM JARVIS Stop Script - Stops all JARVIS processes
echo Stopping JARVIS...

REM Try graceful shutdown via API first
echo Attempting graceful shutdown...
powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/shutdown' -Method POST -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop | Out-Null; Start-Sleep -Seconds 2 } catch {}"

REM Kill by port (more reliable - handles hidden processes)
echo Stopping backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 echo Backend stopped.
)

echo Stopping frontend (port 3000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 echo Frontend stopped.
)

REM Kill by window title to close terminal windows
taskkill /F /FI "WINDOWTITLE eq JARVIS Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq JARVIS Frontend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq launch_jarvis.bat*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq *launch_jarvis*" >nul 2>&1

REM Also kill any remaining Python/Node processes related to JARVIS
taskkill /F /IM python.exe /FI "COMMANDLINE eq *uvicorn*backend.main*" >nul 2>&1
taskkill /F /IM node.exe /FI "COMMANDLINE eq *react-scripts*" >nul 2>&1

echo.
echo JARVIS stopped.
timeout /t 1 >nul
exit
