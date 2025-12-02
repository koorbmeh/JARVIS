# 🤖 JARVIS - Unified Local AI Agent

A single, unified AI agent that combines web search, document memory (RAG), and computer automation into one simple interface. No more switching between different apps!

## ✨ Features

- **💬 Natural Language Interface** - Chat with JARVIS using natural language
- **🔍 Web Search** - Search the web for current information using DuckDuckGo
- **📚 Document Memory (RAG)** - Upload PDFs, TXT, or DOCX files and query them
- **🖥️ Computer Control** - Automate browser actions, take screenshots, type text
- **🧠 Automatic Tool Selection** - JARVIS automatically decides which tool to use
- **🌐 Web Interface** - Clean Gradio interface at http://127.0.0.1:7860

## 📋 Prerequisites

- **Python 3.11+** (Windows Store Python or standard installation)
- **Ollama** installed and running locally
- **Chrome Browser** (for computer automation features)
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

### 3. Verify ChromeDriver

ChromeDriver will be automatically downloaded on first use via `webdriver-manager`. Make sure Chrome is installed.

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
- **Document Query**: "What does my uploaded document say about X?"
- **Computer Control**: "Open browser to YouTube and search for AI News"
- **Screenshot**: "Take a screenshot"
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

- Ensure Chrome is installed
- Close your regular Chrome before using browser automation (or it will use a separate profile)
- ChromeDriver is auto-downloaded on first use

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
- **Browser Automation**: Opens a separate Chrome instance (not your regular Chrome)
- **Document Storage**: Uploaded documents are stored in ChromaDB and persist between sessions

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own use!

## 📄 License

This project is provided as-is for personal use.

---

**Built with**: Python, LangChain, Ollama, Gradio, ChromaDB, Selenium, PyAutoGUI

