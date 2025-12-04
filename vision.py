"""
JARVIS Vision Processing
Image analysis using Ollama vision model with caching
"""

import os
import base64
import hashlib
import json
import requests
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, VISION_TIMEOUT, ENABLE_VISION_CACHE, VISION_CACHE_DIR, USE_CHAINED_MODELS, VISION_MODEL


def _get_image_hash(image_path: str) -> str:
    """Generate hash of image file for caching"""
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def _get_cached_result(image_hash: str, question: str) -> str:
    """Check cache for previous vision result"""
    if not ENABLE_VISION_CACHE or not image_hash:
        return None
    
    try:
        os.makedirs(VISION_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(VISION_CACHE_DIR, f"{image_hash}_{hashlib.md5(question.encode()).hexdigest()}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f).get('result')
    except:
        pass
    
    return None


def _save_to_cache(image_hash: str, question: str, result: str):
    """Save vision result to cache"""
    if not ENABLE_VISION_CACHE or not image_hash:
        return
    
    try:
        os.makedirs(VISION_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(VISION_CACHE_DIR, f"{image_hash}_{hashlib.md5(question.encode()).hexdigest()}.json")
        
        with open(cache_file, 'w') as f:
            json.dump({'result': result, 'question': question}, f)
    except:
        pass


def process_image_with_vision(image_path: str, question: str = "What's in this image?") -> str:
    """Process an image using the vision model via Ollama API with caching"""
    try:
        # Check cache first
        image_hash = _get_image_hash(image_path)
        cached_result = _get_cached_result(image_hash, question)
        if cached_result:
            print("💾 Using cached vision result")
            return cached_result
        
        # Read and encode image as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call Ollama API using chat endpoint (better for vision models)
        ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        # Use chained vision model if enabled, otherwise use integrated VLM
        vision_model = VISION_MODEL if USE_CHAINED_MODELS else OLLAMA_MODEL
        
        prompt = f"{question}\n\nPlease describe what you see in detail."
        
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ],
            "stream": False
        }
        
        response = requests.post(ollama_url, json=payload, timeout=VISION_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            vision_result = result.get("message", {}).get("content", "Could not process image.")
            
            # Cache the result
            _save_to_cache(image_hash, question, vision_result)
            
            return vision_result
        else:
            return f"Error processing image: HTTP {response.status_code}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

