"""
JARVIS Configuration
All configuration constants and settings
"""

# Ollama Configuration
OLLAMA_MODEL = "qwen3-vl:8b-instruct"  # Main LLM model (supports vision)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for RAG (must support embeddings)
OLLAMA_BASE_URL = "http://localhost:11434"

# Directory Configuration
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"
VOICE_MODEL_DIR = "./voice_models"

# Voice Configuration
WHISPER_MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large (tiny is fastest, good for GPU)
RECORD_SECONDS = 3  # Reduced from 5 to 3 seconds for faster processing
USE_FASTER_WHISPER = True  # Use faster-whisper instead of openai-whisper (4-5x faster)

# Agent Configuration
MAX_ITERATIONS = 3  # Allow 3 iterations: 1 for tool call, 1 for follow-up if needed, 1 for final answer
QUICK_RESPONSE_MODE = True  # Skip tools for simple questions (greetings, confirmations, etc.)

# Performance Optimizations
WEB_SEARCH_TIMEOUT = 5  # Seconds to wait for web search
WEB_SEARCH_MAX_CONTENT = 500  # Max words to fetch from web pages (increased to ensure price capture)
VISION_TIMEOUT = 60  # Seconds to wait for vision processing (reduced from 300)
RAG_TOP_K = 2  # Number of document chunks to retrieve (reduced from 3 for speed)

