# JARVIS Launcher for Windows PowerShell
# Double-click this file or run: .\launch_jarvis.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🤖 JARVIS Unified Agent Launcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is running
Write-Host "🔍 Checking Ollama..." -ForegroundColor Yellow
$ollamaRunning = Get-Process ollama -ErrorAction SilentlyContinue

if (-not $ollamaRunning) {
    Write-Host "⚠️  Ollama not running. Starting Ollama..." -ForegroundColor Red
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "✅ Ollama started!" -ForegroundColor Green
} else {
    Write-Host "✅ Ollama is already running" -ForegroundColor Green
}

# Check if model exists
Write-Host "🔍 Checking for qwen3-vl:8b-instruct model..." -ForegroundColor Yellow
$modelCheck = ollama list | Select-String "qwen3-vl:8b-instruct"

if (-not $modelCheck) {
    Write-Host "❌ Model not found!" -ForegroundColor Red
    Write-Host "Please run: ollama pull qwen3-vl:8b-instruct" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
} else {
    Write-Host "✅ Model found!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Launching JARVIS Agent..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 Web interface will open at: http://127.0.0.1:7860" -ForegroundColor Yellow
Write-Host "📍 Press Ctrl+C to stop the agent" -ForegroundColor Yellow
Write-Host ""

# Launch the agent
python jarvis_agent.py

# If the script exits
Write-Host ""
Write-Host "👋 JARVIS stopped." -ForegroundColor Cyan
Read-Host "Press Enter to close"
