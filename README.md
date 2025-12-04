# 🤖 JARVIS - Unified Local AI Agent

A single, unified AI agent that combines web search, document memory (RAG), and computer automation into one simple interface. No more switching between different apps!

## ✨ Features

- **💬 Natural Language Interface** - Chat with JARVIS using natural language
- **👁️ Vision Capabilities** - Upload images directly in the chat for JARVIS to analyze
- **🎤 Voice Input & Output** - Speak to JARVIS and get voice responses (Whisper STT + TTS)
- **🔍 Web Search** - Search the web for current information using DuckDuckGo
- **📚 Document Memory (RAG)** - Upload PDFs, TXT, or DOCX files and query them
- **💾 Conversation Logging** - All conversations are automatically saved and searchable via RAG for future reference
- **🖥️ Computer Control** - Automate browser actions, find and click text, take screenshots, type text
- **🧠 Automatic Tool Selection** - JARVIS automatically decides which tool to use
- **🌐 Modern Web Interface** - FastAPI + React interface with Cursor-like design at http://localhost:3000

## 📋 Prerequisites

- **Python 3.11+** (Windows Store Python or standard installation)
- **Node.js 16+** - Install from https://nodejs.org/ (required for React frontend)
- **Ollama** installed and running locally
- **Default web browser** (for opening URLs - uses your system's default browser)
- **qwen2.5:7b model** in Ollama (fast text/reasoning model for chained approach)
- **qwen3-vl:4b model** in Ollama (lightweight vision model for chained approach)
- **nomic-embed-text model** in Ollama (for document embeddings/RAG)

## 🚀 Installation

### 1. Install Ollama and Models

Download and install [Ollama](https://ollama.ai/), then pull the required models:

```powershell
# Fast text model (for chained approach) - REQUIRED
ollama pull qwen2.5:7b

# Lightweight vision model (for chained approach) - REQUIRED
ollama pull qwen3-vl:4b

# Embedding model for RAG (document memory) - REQUIRED
ollama pull nomic-embed-text
```

**Optional - Alternative Integrated VLM Approach:**
If you prefer to use a single integrated vision-language model instead of the chained approach, you can use `qwen3-vl:8b-instruct`:

```powershell
# Integrated VLM (alternative to chained models - OPTIONAL)
ollama pull qwen3-vl:8b-instruct
```

Then set `USE_CHAINED_MODELS = False` in `config.py` and ensure `OLLAMA_MODEL = "qwen3-vl:8b-instruct"`.

**Note**: The chained approach (default) is 2-3x faster for most tasks. The integrated VLM is simpler but slower.

Verify the models are installed:

```powershell
ollama list
```

For the default chained setup, you should see `qwen2.5:7b`, `qwen3-vl:4b`, and `nomic-embed-text`.

### 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

**Note**: On Windows, `pywin32` is recommended (but optional) for better multi-monitor support when taking screenshots. It's included in requirements.txt but will only install on Windows.

### 4. (Optional) Install Tesseract OCR for Text Finding

For optimal text finding in screenshots, install Tesseract OCR:

- **Windows**: 
  - **Recommended**: Download the installer from [GitHub - UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - During installation, check the option to "Add to PATH" or manually add `C:\Program Files\Tesseract-OCR` to your system PATH
  - After installation, restart your terminal/PowerShell
  - **Alternative**: If you have Chocolatey installed: `choco install tesseract`
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**Note**: If Tesseract is not installed or not in PATH, JARVIS will automatically fall back to using the vision model to locate text, which may be less accurate but will still work. The vision model fallback is fully functional and will attempt to find and click text based on visual analysis.

### 5. (Optional) Voice Capabilities

For voice input and output, install the voice dependencies:

```powershell
pip install openai-whisper pyaudio soundfile pyttsx3
```

**Note**: 
- **Whisper** (STT): The first run will download the model (~150MB for "base" model). Larger models ("small", "medium") provide better accuracy but require more disk space and processing time.
- **pyttsx3** (TTS): Uses your system's built-in voices. On Windows, you can change voices in Settings > Time & Language > Speech.
- **pyaudio**: May require additional system dependencies on some platforms. If installation fails, see [PyAudio installation guide](https://people.csail.mit.edu/hubert/pyaudio/docs/).

Voice features are optional - JARVIS works perfectly fine with text-only input.

### 6. Ready to Go!

JARVIS uses your system's default browser to open URLs, so no additional browser setup is needed.

## 🎯 Quick Start

### Option 1: Double-Click Launcher (Windows - Recommended)

Simply double-click `launch_jarvis.bat` to start JARVIS. The script will:
- Start the FastAPI backend (http://127.0.0.1:8000)
- Start the React frontend (http://localhost:3000)
- Wait for both to be ready
- Open your browser automatically

### Option 2: Manual Start

**Terminal 1 - Backend:**
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm start
```

Then open http://localhost:3000 in your browser.


## 📖 Usage

### Web Interface

1. Open your browser to http://localhost:3000 (or it will open automatically)
2. Use the chat interface to interact with JARVIS
3. Upload images by dragging and dropping or pasting from clipboard
4. Click the microphone button for voice input
5. Toggle "Voice Responses" for text-to-speech output
6. Use the "Close JARVIS" button to shut down all services

### Example Commands

- **Web Search**: "What's the current spot price of silver?"
- **Vision**: Paste an image in the chat and ask "What's in this image?" or "What does this chart show?"
- **Document Query**: "What does my uploaded document say about X?"
- **Conversation History**: "What did we discuss about silver prices yesterday?" (JARVIS automatically loads recent conversation logs on startup and searches them via RAG)
- **File Management**: 
  - "Create a file called notes.txt with the content: Meeting notes..."
  - "Read the file notes.txt"
  - "List all my files"
  - Created files are automatically added to RAG (if .txt, .pdf, or .docx)
- **Computer Control**: 
  - "Open browser to YouTube and search for AI News"
  - "Open Google Sheets" (correctly opens sheets.google.com)
  - "Open Google Docs" (correctly opens docs.google.com)
  - "Take a screenshot and find the text 'Submit'"
- **Voice**: 
  - Click the microphone button and speak your message (Voice Input mode)
  - Toggle "Conversation Mode" for voice responses (STT + TTS)
  - Speak naturally - JARVIS uses Whisper for accurate transcription
- **Combined**: "Search for EV prices under $50k and open the first result"

## ⚡ Performance Optimization

JARVIS uses the chained models approach by default for 2-3x faster performance. If you're experiencing slowness or want to optimize further, here are proven fixes:

### 1. Chained Models Approach (Default - 2-3x Faster)

**The Problem**: Integrated VLMs like `qwen3-vl:8b-instruct` process vision even for text queries, adding 20-40% overhead. This makes them slower for most JARVIS tasks (80% are text/tool queries, only 20% need vision).

**The Solution**: Chain a fast text model with a lightweight vision model. This gives:
- **50-80 t/s for text/tools** (vs 30-50 t/s with integrated VLM)
- **5-15s average response** (vs 15-30s with integrated VLM)
- **Vision only when needed** (adds ~0.3-0.5s per image)

**Default Configuration**:
The chained approach is enabled by default (`USE_CHAINED_MODELS = True` in `config.py`). Simply pull the required models (see Installation section):
- `qwen2.5:7b` - Text/reasoning core (fast for tools, ~4GB VRAM)
- `qwen3-vl:4b` - Lightweight vision (on-demand, ~2GB VRAM)

**For Larger GPUs (16GB+ VRAM)**:
You can use `deepseek-r1:32b` as the text model for even better reasoning:
```python
USE_CHAINED_MODELS = True
TEXT_MODEL = "deepseek-r1:32b"  # Requires 20GB+ VRAM
VISION_MODEL = "qwen3-vl:4b"
```

**Expected gains**: 2-3x faster responses (5-15s instead of 15-30s for text queries)

### 2. Force Full GPU Offload

Set environment variable before running JARVIS to offload all layers to GPU:

**PowerShell:**
```powershell
$env:OLLAMA_NUM_GPU_LAYERS="999"
launch_jarvis.bat
```

**Bash:**
```bash
export OLLAMA_NUM_GPU_LAYERS=999
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Hardware Optimizations

- **Close background apps** (Chrome tabs eat VRAM) - use `nvidia-smi` to monitor GPU usage
- **Aim for <80% VRAM usage** - if over, use a smaller quantized model
- **Vision caching is enabled by default** - screenshots are cached to avoid redundant OCR

### 4. Advanced: Custom Model Selection

The chained models approach is already fully implemented in the codebase. You can customize which models to use by editing `config.py`:

- **For RTX 2060 (6GB VRAM)**: Use `qwen2.5:7b` + `qwen3-vl:4b` (default in config)
- **For larger GPUs (16GB+ VRAM)**: You can use `deepseek-r1:32b` as the text model for better reasoning:
  ```python
  USE_CHAINED_MODELS = True
  TEXT_MODEL = "deepseek-r1:32b"  # Requires 20GB+ VRAM
  VISION_MODEL = "qwen3-vl:4b"
  ```

**Expected gains**: 2-3x speed improvement on RTX 2060 (5-15s for text, +0.5s for vision)

### 5. Alternative: Use llama.cpp Server (Optional)

For maximum GPU utilization on older cards, you can use llama.cpp server instead of Ollama:

1. Build llama.cpp with CUDA: `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j CUDA=1`
2. Download GGUF models from HuggingFace
3. Serve: `./llama-server -m model.gguf --n-gpu-layers 999 -c 8192 --port 11435`
4. Update `config.py`: `OLLAMA_BASE_URL = "http://localhost:11435"`

**Expected gains**: 2-3x speed improvement on RTX 2060 (10-15 t/s routine)

### 6. Future Hardware Upgrades

On RTX 5070Ti (16GB VRAM) or similar:
- Full Q5/Q6 quantization at 40-60 t/s
- No VRAM spills to CPU/RAM
- JARVIS will feel instant

## 🛠️ Configuration

Edit `config.py` to customize:

- **Chained Models**: `USE_CHAINED_MODELS = True` (default: enabled - uses `qwen2.5:7b` + `qwen3-vl:4b`)
  - Set to `False` to use integrated VLM (`qwen3-vl:8b-instruct`) instead
- **Text Model**: `TEXT_MODEL = "qwen2.5:7b"` (default: for chained approach)
- **Vision Model**: `VISION_MODEL = "qwen3-vl:4b"` (default: for chained approach)
- **Ollama LLM Model**: `OLLAMA_MODEL = "qwen3-vl:8b-instruct"` (used when `USE_CHAINED_MODELS = False`)
- **Ollama Embedding Model**: `OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"` (default: for RAG)
- **Ollama URL**: `OLLAMA_BASE_URL = "http://localhost:11434"` (default)
- **ChromaDB Directory**: `CHROMA_DB_DIR = "./chroma_db"` (default)
- **Vision Caching**: `ENABLE_VISION_CACHE = True` (default: enabled for faster screenshot processing)
- **Conversation Logging**: `ENABLE_CONVERSATION_LOGGING = True` (default: enabled - saves all conversations to `documents/conversation_logs/`)
- **Auto-add to RAG**: `AUTO_ADD_CONVERSATIONS_TO_RAG = True` (default: enabled - automatically makes conversation logs searchable)
- **Load History on Startup**: `LOAD_CONVERSATION_HISTORY_ON_STARTUP = True` (default: enabled - loads recent conversation logs on startup for context)
- **History Days**: `CONVERSATION_HISTORY_DAYS = 7` (default: 7 days - number of days of conversation history to load on startup)

## 📁 Project Structure

```
JARVIS/
├── backend/
│   └── main.py              # FastAPI backend server
├── frontend/
│   ├── src/                 # React source files
│   │   ├── components/      # React components
│   │   ├── App.js           # Main app component
│   │   └── index.js         # Entry point
│   ├── public/              # Static files
│   └── package.json         # Node.js dependencies
├── requirements.txt         # Python dependencies
├── launch_jarvis.bat        # Windows launcher (double-click)
├── stop_jarvis.bat          # Stop script
├── README.md                # This file
├── chroma_db/               # Vector database (auto-created)
├── documents/               # Uploaded documents (auto-created)
│   └── conversation_logs/   # Conversation history logs (auto-created, searchable via RAG)
├── vision_cache/            # Cached vision results (auto-created)
└── debug_logs/              # Debug logs (auto-created)
```

## 🔧 Troubleshooting

### Web Search Returns No Results

Web search uses DuckDuckGo HTML interface. If you get no results:
- Try simpler search queries (2-5 words)
- Check your internet connection
- The search may be rate-limited (wait a few minutes)

### Browser Automation Fails

- JARVIS uses your system's default browser
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

### Verifying GPU Usage

**Ollama (LLM & Embeddings):**
- Ollama automatically uses GPU if available
- Check GPU usage: `ollama ps` (shows running models and GPU memory)
- Check Ollama logs for GPU initialization messages
- If GPU isn't being used, ensure CUDA drivers are installed and Ollama was built with GPU support

**Whisper (Speech-to-Text):**
- JARVIS automatically detects and uses GPU for Whisper if available
- Check console output on startup - should show "🎮 GPU detected: [GPU Name]" if using GPU
- If using CPU, transcription will be slower but still functional

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

- **Response Time**: 5-15 seconds per query is typical (local processing on RTX 2060 with chained models)
- **VRAM Usage**: ~6GB / 6GB with chained models (qwen2.5:7b + qwen3-vl:4b), or ~5.5GB with integrated VLM (qwen3-vl:8b-instruct)
- **Browser Automation**: Opens URLs in new tabs in your default browser
- **Document Storage**: Uploaded documents are stored in ChromaDB and persist between sessions
- **Shutdown**: Use the "Close JARVIS" button in the web interface or run `stop_jarvis.bat` to shut down all services

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own use!

## 📄 License

This project is provided as-is for personal use.

---

**Built with**: Python, FastAPI, React, LangChain, Ollama, ChromaDB, PyAutoGUI

