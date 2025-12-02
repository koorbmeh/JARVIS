# 🤖 JARVIS - Unified Local AI Agent

A single, unified AI agent that combines web search, document memory (RAG), and computer automation into one simple interface. No more switching between different apps!

## ✨ Features

- **💬 Natural Language Interface** - Chat with JARVIS using natural language
- **👁️ Vision Capabilities** - Upload images directly in the chat for JARVIS to analyze
- **🔍 Web Search** - Search the web for current information using DuckDuckGo
- **📚 Document Memory (RAG)** - Upload PDFs, TXT, or DOCX files and query them
- **🖥️ Computer Control** - Automate browser actions, find and click text, take screenshots, type text
- **🧠 Automatic Tool Selection** - JARVIS automatically decides which tool to use
- **🌐 Unified Web Interface** - Clean Gradio interface with unified text/image input at http://127.0.0.1:7860

## 📋 Prerequisites

- **Python 3.11+** (Windows Store Python or standard installation)
- **Ollama** installed and running locally
- **Default web browser** (for opening URLs - uses your system's default browser)
- **qwen3-vl:8b-instruct model** in Ollama

## 🚀 Installation

### 1. Install Ollama and Model

Download and install [Ollama](https://ollama.ai/), then pull the required model:

```powershell
ollama pull qwen3-vl:8b-instruct
```

Verify the model is installed:

```powershell
ollama list
```

### 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: On Windows, `pywin32` is recommended (but optional) for better multi-monitor support when taking screenshots. It's included in requirements.txt but will only install on Windows.

### 3. (Optional) Install Tesseract OCR for Text Finding

For optimal text finding in screenshots, install Tesseract OCR:

- **Windows**: 
  - **Recommended**: Download the installer from [GitHub - UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - During installation, check the option to "Add to PATH" or manually add `C:\Program Files\Tesseract-OCR` to your system PATH
  - After installation, restart your terminal/PowerShell
  - **Alternative**: If you have Chocolatey installed: `choco install tesseract`
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**Note**: If Tesseract is not installed or not in PATH, JARVIS will automatically fall back to using the vision model to locate text, which may be less accurate but will still work. The vision model fallback is fully functional and will attempt to find and click text based on visual analysis.

### 4. Ready to Go!

JARVIS uses your system's default browser to open URLs, so no additional browser setup is needed.

## 🎯 Quick Start

### Option 1: Double-Click Launcher (Windows)

Simply double-click `launch_jarvis.bat` to start JARVIS.

### Option 2: Command Line

```powershell
python jarvis_agent.py
```

The web interface will be available at: **http://127.0.0.1:7860**

## 📖 Usage

### Web Interface

1. Open your browser to http://127.0.0.1:7860
2. Use the **Chat** tab to interact with JARVIS
3. Use the **Documents** tab to upload files for RAG
4. Use the **Info** tab for tips and examples

### Example Commands

- **Web Search**: "What's the current spot price of silver?"
- **Vision**: Paste an image in the chat and ask "What's in this image?" or "What does this chart show?"
- **Document Query**: "What does my uploaded document say about X?"
- **Computer Control**: 
  - "Open browser to YouTube and search for AI News"
  - "Open Google Sheets and click on the word 'bills'"
  - "Take a screenshot and find the text 'Submit'"
- **Combined**: "Search for EV prices under $50k and open the first result"

## 🛠️ Configuration

Edit `jarvis_agent.py` to customize:

- **Ollama Model**: Change `OLLAMA_MODEL` (default: `qwen3-vl:8b-instruct`)
- **Ollama URL**: Change `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- **ChromaDB Directory**: Change `CHROMA_DB_DIR` (default: `./chroma_db`)

## 📁 Project Structure

```
JARVIS/
├── jarvis_agent.py          # Main application
├── requirements.txt          # Python dependencies
├── launch_jarvis.bat        # Windows launcher (double-click)
├── launch_jarvis.ps1        # PowerShell launcher (with Ollama checks)
├── README.md                 # This file
├── chroma_db/                # Vector database (auto-created)
└── documents/                # Uploaded documents (auto-created)
```

## 🔧 Troubleshooting

### Web Search Returns No Results

Web search uses DuckDuckGo HTML interface. If you get no results:
- Try simpler search queries (2-5 words)
- Check your internet connection
- The search may be rate-limited (wait a few minutes)

### Browser Automation Fails

- JARVIS uses your system's default browser (the same one showing the Gradio interface)
- URLs open in new tabs in your existing browser
- No ChromeDriver or Selenium setup needed

### Multi-Display Setup Issues

If you have multiple monitors and screenshots capture the wrong screen:
- **Windows**: Install `pywin32` for better window detection: `pip install pywin32`
- JARVIS will automatically try to focus and capture only the browser window
- If issues persist, make sure the browser window is active/foreground before taking screenshots

### Ollama Connection Errors

- Verify Ollama is running: `ollama list`
- Check the model is installed: `ollama list | findstr qwen3-vl`
- Ensure `OLLAMA_BASE_URL` matches your Ollama instance

### Deprecation Warnings

The code uses deprecated LangChain imports for compatibility. These warnings are safe to ignore. The functionality works correctly.

## 🎓 How It Works

JARVIS uses a **LangChain ReAct agent** that:

1. **Receives your query** in natural language
2. **Decides which tool to use** (WebSearch, DocumentQuery, or ComputerControl)
3. **Executes the tool** and gets results
4. **Formats the response** using the LLM

The agent follows a "Thought → Action → Observation" loop until it has enough information to provide a final answer.

## 📝 Notes

- **Response Time**: 30-60 seconds per query is normal (local processing on RTX 2060)
- **VRAM Usage**: ~5.5GB / 6GB with qwen3-vl:8b-instruct
- **Browser Automation**: Opens URLs in new tabs in your default browser (same browser as the Gradio interface)
- **Document Storage**: Uploaded documents are stored in ChromaDB and persist between sessions

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own use!

## 📄 License

This project is provided as-is for personal use.

---

**Built with**: Python, LangChain, Ollama, Gradio, ChromaDB, PyAutoGUI

