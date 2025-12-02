"""
JARVIS Vision Processing
Image analysis using Ollama vision model
"""

import base64
import requests
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, VISION_TIMEOUT


def process_image_with_vision(image_path: str, question: str = "What's in this image?") -> str:
    """Process an image using the vision model via Ollama API"""
    try:
        # Read and encode image as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call Ollama API using chat endpoint (better for vision models)
        ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        prompt = f"{question}\n\nPlease describe what you see in detail."
        
        payload = {
            "model": OLLAMA_MODEL,
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
            return result.get("message", {}).get("content", "Could not process image.")
        else:
            return f"Error processing image: HTTP {response.status_code}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

