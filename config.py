"""
JARVIS Configuration
All configuration constants and settings
"""

# Ollama Configuration
# PERFORMANCE TIP: For RTX 2060 (6GB VRAM) or similar, consider:
#   - Current: qwen3-vl:8b-instruct (integrated VLM, ~5.5GB VRAM, slower for text-only queries)
#   - Alternative: Chained approach (see README) - DeepSeek-R1 (text) + lightweight vision model
#     This gives 2-3x speed for text/tool queries (50-80 t/s vs 30-50 t/s) with vision on-demand
#   - Check available models: `ollama list` or visit https://ollama.com/library
OLLAMA_MODEL = "qwen3-vl:8b-instruct"  # Main LLM model (supports vision)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for RAG (must support embeddings)
OLLAMA_BASE_URL = "http://localhost:11434"

# Chained Model Configuration (Optional - for faster text/tool performance)
# If enabled, uses fast text model + lightweight vision model only when needed
# IMPORTANT: For RTX 2060 (6GB VRAM), use SMALL models (7b or less)!
USE_CHAINED_MODELS = True  # Set to True to enable chained approach (see README for setup)

# Text model options for 6GB VRAM (pick one):
# - "qwen2.5:7b" (recommended: fast, good quality, non-quantized, ~4GB VRAM) ← CURRENT
# - "qwen2.5:7b-instruct-q4_K_M" (quantized version - may be slower on some systems, ~4.7GB VRAM) ← Tried, reverted (slower)
# - "qwen2.5:3b" (ultra-fast: ~2GB VRAM, slightly less capable) ← If 7b is too slow
# - "llama3.2:3b" (very fast: ~2GB VRAM, good for simple tasks)
# - "phi3:3.8b" (fast: ~2.5GB VRAM, Microsoft's efficient model)
# - "ministral-3:8b-instruct-2512-q4_K_M" (excellent reasoning, multimodal capable, ~4-5GB VRAM) ← Tried, reverted
# - "deepseek-r1:7b" (if available: reasoning-focused, ~4GB VRAM)
# AVOID: deepseek-r1:32b (needs 20GB+ VRAM - too large for 6GB cards!)
TEXT_MODEL = "qwen2.5:7b"  # Text/reasoning model (fits 6GB VRAM, ~4GB VRAM usage, non-quantized - faster than quantized version)
VISION_MODEL = "qwen3-vl:4b"  # Lightweight vision model (on-demand only)

# GPU Optimization (set environment variable before running JARVIS)
# PowerShell: $env:OLLAMA_NUM_GPU_LAYERS="999"
# Bash: export OLLAMA_NUM_GPU_LAYERS=999
# This forces Ollama to offload all layers to GPU (if VRAM allows)

# Directory Configuration
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"
VOICE_MODEL_DIR = "./voice_models"

# Voice Configuration
WHISPER_MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large (tiny is fastest, good for GPU)
RECORD_SECONDS = 3  # Reduced from 5 to 3 seconds for faster processing
USE_FASTER_WHISPER = True  # Use faster-whisper instead of openai-whisper (4-5x faster)

# Agent Configuration
MAX_ITERATIONS = 10  # Allow up to 10 iterations for complex questions (increased from 3)
QUICK_RESPONSE_MODE = True  # Skip tools for simple questions (greetings, confirmations, etc.)

# Performance Optimizations
WEB_SEARCH_TIMEOUT = 30  # Seconds to wait for web search (increased from 5 for complex queries)
WEB_SEARCH_MAX_CONTENT = 500  # Max words to fetch from web pages (increased to ensure price capture)
VISION_TIMEOUT = 300  # Seconds to wait for vision processing (increased from 60 for complex images)
RAG_TOP_K = 2  # Number of document chunks to retrieve (reduced from 3 for speed)

# Vision Caching (reduces redundant OCR/vision processing)
ENABLE_VISION_CACHE = True  # Cache vision results by image hash
VISION_CACHE_DIR = "./vision_cache"  # Directory for cached vision results

# Conversation Logging (saves chat history for RAG reference)
ENABLE_CONVERSATION_LOGGING = True  # Save conversations to files
AUTO_ADD_CONVERSATIONS_TO_RAG = True  # Automatically add conversation logs to RAG for future reference
LOAD_CONVERSATION_HISTORY_ON_STARTUP = True  # Load recent conversation logs on startup for context
CONVERSATION_HISTORY_DAYS = 7  # Number of days of conversation history to load on startup (default: 7 days)

# Self-Reflection (learns from errors and slow responses)
ENABLE_SELF_REFLECTION = True  # Log errors/slow responses and generate improvement examples
SLOW_RESPONSE_THRESHOLD = 15.0  # Seconds - responses slower than this trigger reflection
