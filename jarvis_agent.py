"""
JARVIS - Unified Local AI Agent (Compatibility Version)
Works with existing package versions - no upgrades needed
"""

import os
import json
import warnings
import re
import urllib.parse
from typing import Optional, List
import gradio as gr

# Suppress deprecation warnings for Ollama imports
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Use deprecated imports (compatible with langchain 0.3.27 and langchain-core 0.3.80)
# langchain-ollama requires langchain-core>=1.0.0 which conflicts with langchain 0.3.27
from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import pyautogui
import webbrowser
import time
import base64
from PIL import Image
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    # Try to install pywin32 if not available (optional)
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
        import win32gui
        import win32con
        WIN32_AVAILABLE = True
    except:
        WIN32_AVAILABLE = False
        print("⚠️  pywin32 not available - screenshots will capture all displays. Install with: pip install pywin32")
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
    # Configure pytesseract to use the default Windows installation path if not in PATH
    import os
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not available - text finding in screenshots will use vision model only")

# ==================== CONFIGURATION ====================

OLLAMA_MODEL = "qwen3-vl:8b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"

# No longer needed - using webbrowser module instead of Selenium

# ==================== INITIALIZE OLLAMA ====================

print("🧠 Initializing Ollama connection...")
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7
)

embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

print(f"✅ Connected to Ollama: {OLLAMA_MODEL}")

# ==================== VISION PROCESSING ====================

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
        
        response = requests.post(ollama_url, json=payload, timeout=300)  # Increased timeout for vision processing
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "Could not process image.")
        else:
            return f"Error processing image: HTTP {response.status_code}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# ==================== RAG SETUP (Document Memory) ====================

class DocumentMemory:
    def __init__(self):
        self.vectorstore = None
        self.documents_loaded = False
        
    def add_documents(self, file_paths: List[str]):
        """Add documents to the vector database"""
        from langchain_community.document_loaders import (
            TextLoader, 
            PyPDFLoader,
            Docx2txtLoader
        )
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                    
                docs = loader.load()
                documents.extend(text_splitter.split_documents(docs))
                print(f"📄 Added: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if documents:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR
                )
            else:
                self.vectorstore.add_documents(documents)
            
            self.documents_loaded = True
            print(f"✅ {len(documents)} document chunks indexed")
        else:
            print("⚠️ No documents loaded")
    
    def query(self, question: str) -> str:
        """Query the document database"""
        if not self.documents_loaded or self.vectorstore is None:
            return "No documents have been added yet. Please upload documents first."
        
        try:
            # Simple retrieval without RetrievalQA chain
            docs = self.vectorstore.similarity_search(question, k=3)
            
            if not docs:
                return "No relevant information found in documents."
            
            # Combine context and ask the LLM
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Based on the following context from documents, answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error querying documents: {str(e)}"

doc_memory = DocumentMemory()

# ==================== TOOLS ====================

def web_search_tool(query: str) -> str:
    """Search the web for current information using DuckDuckGo HTML interface"""
    try:
        print(f"🔍 Searching web for: {query}")
        
        # Use direct HTTP request to DuckDuckGo HTML interface (works when DDGS package fails)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Error: DuckDuckGo returned status code {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find search results - DuckDuckGo HTML structure
        results = []
        result_divs = soup.find_all('div', class_='result')
        
        for div in result_divs[:5]:  # Limit to 5 results
            title_elem = div.find('a', class_='result__a')
            snippet_elem = div.find('a', class_='result__snippet')
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                url_link = title_elem.get('href', 'No URL')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description available"
                
                # Clean up URL (remove DuckDuckGo redirect)
                if url_link.startswith('/l/?kh='):
                    # Extract actual URL from DuckDuckGo redirect
                    try:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url_link).query)
                        if 'uddg' in parsed:
                            url_link = urllib.parse.unquote(parsed['uddg'][0])
                    except:
                        pass
                
                results.append({
                    'title': title,
                    'body': snippet,
                    'url': url_link
                })
        
        if not results:
            return "No results found. Try a different search query."
        
        # Skip ad/redirect URLs and prioritize reputable financial sites
        reputable_domains = ['kitco.com', 'apmex.com', 'jmbullion.com', 'sdbullion.com', 
                            'moneymetals.com', 'goldprice.org', 'silverprice.org',
                            'bloomberg.com', 'reuters.com', 'marketwatch.com']
        
        # Try to fetch actual content from reputable sites first
        for result in results:
            try:
                url = result['url']
                # Skip ad/redirect URLs
                if not url or not url.startswith('http'):
                    continue
                if 'duckduckgo.com/y.js' in url or 'bing.com/aclick' in url:
                    continue
                
                # Check if it's a reputable domain
                is_reputable = any(domain in url.lower() for domain in reputable_domains)
                
                # Prioritize reputable sites, but also try others if no reputable ones found
                if is_reputable or len([r for r in results if any(d in r.get('url', '').lower() for d in reputable_domains)]) == 0:
                    print(f"   Fetching content from: {url}")
                    page_response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                    if page_response.status_code == 200:
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                        # Extract text content (remove scripts, styles)
                        for script in page_soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        page_text = page_soup.get_text(separator=' ', strip=True)
                        # Get more content for price extraction (first 500 words)
                        page_text = ' '.join(page_text.split()[:500])
                        result['page_content'] = page_text
                        
                        # Try to extract price from content
                        # Look for price patterns like $XX.XX per ounce
                        price_patterns = [
                            r'\$[\d,]+\.?\d*\s*(?:per\s+)?(?:ounce|oz|troy\s+ounce)',
                            r'(?:spot\s+price|current\s+price|price)[:\s]+\$?[\d,]+\.?\d*',
                            r'\$[\d,]+\.?\d*\s*(?:USD|US\s+dollar)',
                        ]
                        for pattern in price_patterns:
                            matches = re.findall(pattern, page_text, re.IGNORECASE)
                            if matches:
                                result['extracted_price'] = matches[0]
                                break
                        
                        # Only fetch from one reputable site to avoid rate limiting
                        if is_reputable:
                            break
            except Exception as e:
                print(f"   Could not fetch {result['url']}: {e}")
                result['page_content'] = None
        
        output = f"Web search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   {result['body']}\n"
            if 'extracted_price' in result and result['extracted_price']:
                output += f"   EXTRACTED PRICE: {result['extracted_price']}\n"
            if 'page_content' in result and result['page_content']:
                # Include relevant excerpt from page (first 400 chars)
                content = result['page_content'][:400]
                output += f"   Page content: {content}...\n"
            output += f"   URL: {result['url']}\n\n"
        
        return output
    except Exception as e:
        return f"Error searching web: {str(e)}\nTip: Try simpler search terms."

def document_query_tool(query: str) -> str:
    """Query uploaded documents using RAG"""
    print(f"📚 Querying documents: {query}")
    return doc_memory.query(query)

def get_browser_window_region():
    """Get the region (coordinates) of the browser window for focused screenshot"""
    if not WIN32_AVAILABLE:
        return None
    
    try:
        def enum_handler(hwnd, results):
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            # Look for common browser windows - prioritize windows with actual content
            if any(browser in window_text.lower() or browser in class_name.lower() 
                   for browser in ['chrome', 'firefox', 'edge', 'brave', 'opera', 'safari']):
                if win32gui.IsWindowVisible(hwnd):
                    # Get window rectangle
                    rect = win32gui.GetWindowRect(hwnd)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    # Only include reasonably sized windows (not minimized)
                    if width > 200 and height > 200:
                        results.append((hwnd, window_text, rect))
        
        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        
        if windows:
            # Get the most recently used browser window
            hwnd, window_text, rect = windows[0]
            # Bring window to foreground
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)  # Give window time to come to foreground
            return rect  # Returns (left, top, right, bottom)
    except Exception as e:
        print(f"   Could not find browser window: {e}")
    
    return None

def focus_browser_window():
    """Try to focus/activate the browser window before taking a screenshot (for multi-display setups)"""
    region = get_browser_window_region()
    return region is not None

def computer_control_tool(action: str) -> str:
    """
    Control the computer. Supported actions:
    - 'open browser: <url>' - Opens a browser to URL (browser stays open)
    - 'open browser: youtube.com/search?q=<query>' - Opens YouTube and searches
    - 'screenshot' - Takes a screenshot
    - 'find_text_and_click: <text>' - Takes screenshot, finds text using OCR/vision, and clicks it
    - 'type: <text>' - Types text
    - 'click: <x>,<y>' - Clicks at coordinates
    """
    try:
        print(f"🖥️ Computer control: {action}")
        
        if action.startswith("open browser:"):
            url = action.replace("open browser:", "").strip()
            
            # Handle YouTube search requests - check if user wants to search YouTube
            if "youtube" in url.lower():
                # Check if there's a search request
                if "search for" in url.lower():
                    # Extract search query using regex
                    search_match = re.search(r'search for ["\']([^"\']+)["\']', url, re.IGNORECASE)
                    if not search_match:
                        # Try without quotes
                        search_match = re.search(r'search for\s+([^\s]+(?:\s+[^\s]+)*)', url, re.IGNORECASE)
                    
                    if search_match:
                        query = search_match.group(1).strip()
                        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
                    else:
                        # Fallback: just open YouTube homepage
                        url = "https://www.youtube.com"
                else:
                    # Just open YouTube homepage
                    url = "https://www.youtube.com"
            
            # Ensure URL has protocol (only if it doesn't already have one)
            if not url.startswith(('http://', 'https://')):
                # Check if it looks like a domain name
                if '.' in url and not url.startswith('/'):
                    url = 'https://' + url
                else:
                    return f"Error: Invalid URL format: {url}"
            
            # Use Python's built-in webbrowser module to open in default browser
            # This opens a new tab in the user's default browser (same as the Gradio interface)
            try:
                webbrowser.open(url, new=2)  # new=2 opens in a new tab if possible
                # Give the browser a moment to start loading
                time.sleep(1)
                return f"Opened {url} in your default browser (new tab)."
            except Exception as e:
                return f"Error opening browser: {str(e)}"
        
        elif action == "screenshot":
            # Try to get browser window region and focus it (for multi-display setups)
            browser_region = get_browser_window_region()
            
            if browser_region:
                # Capture only the browser window region
                left, top, right, bottom = browser_region
                screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
            else:
                # Fallback to full screen
                screenshot = pyautogui.screenshot()
            
            screenshot_path = "screenshot.png"
            screenshot.save(screenshot_path)
            return f"Screenshot saved to {screenshot_path}. You can now ask me to analyze it by uploading it or asking 'What's in the screenshot?'"
        
        elif action.startswith("find_text_and_click:"):
            # Find text in screenshot and click it
            search_text = action.replace("find_text_and_click:", "").strip().lower()
            
            # Wait a moment for page to load if browser was just opened
            time.sleep(2)
            
            # Try to get browser window region and focus it (for multi-display setups)
            browser_region = get_browser_window_region()
            window_offset_x = 0
            window_offset_y = 0
            
            if browser_region:
                # Capture only the browser window region
                left, top, right, bottom = browser_region
                window_offset_x = left  # Save offset for coordinate conversion
                window_offset_y = top
                screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
            else:
                # Fallback to full screen
                screenshot = pyautogui.screenshot()
            
            screenshot_path = "screenshot_temp.png"
            screenshot.save(screenshot_path)
            
            # Try OCR first (more accurate coordinates)
            if TESSERACT_AVAILABLE:
                try:
                    # Check if Tesseract is actually available (not just the Python package)
                    try:
                        pytesseract.get_tesseract_version()
                    except Exception as tesseract_check_error:
                        raise Exception(f"Tesseract OCR not found in PATH: {tesseract_check_error}. Please install Tesseract OCR (see README).")
                    
                    # Use pytesseract to find text with coordinates
                    data = pytesseract.image_to_data(screenshot, output_type=Output.DICT)
                    
                    # Find the text in OCR results
                    found = False
                    for i, text in enumerate(data['text']):
                        if search_text in text.lower():
                            # Coordinates are relative to the screenshot, convert to screen coordinates
                            x = window_offset_x + data['left'][i] + data['width'][i] // 2  # Center of text
                            y = window_offset_y + data['top'][i] + data['height'][i] // 2
                            print(f"   Clicking at screen coordinates ({x}, {y}) - text found at relative ({data['left'][i]}, {data['top'][i]})")
                            # Ensure browser window is focused before clicking
                            if browser_region:
                                time.sleep(0.3)  # Give window time to be fully focused
                            pyautogui.click(x, y)
                            found = True
                            return f"Found and clicked on '{search_text}' at screen coordinates ({x}, {y})"
                    
                    if not found:
                        # Fallback to vision model
                        vision_response = process_image_with_vision(
                            screenshot_path, 
                            question=f"Where is the word '{search_text}' located on the screen? Describe its position in detail, including approximate pixel coordinates if possible, or describe its location relative to other elements."
                        )
                        return f"Could not find '{search_text}' using OCR. Vision analysis: {vision_response}. You may need to click manually or provide more specific location."
                except Exception as ocr_error:
                    print(f"   OCR error: {ocr_error}, falling back to vision model")
                    # Fallback to vision model with retry logic
                    try:
                        vision_response = process_image_with_vision(
                            screenshot_path, 
                            question=f"Where is the word '{search_text}' located on the screen? Describe its position in detail, including approximate pixel coordinates (x, y) if you can estimate them."
                        )
                        # Try to extract coordinates from vision response
                        coord_match = re.search(r'\((\d+),\s*(\d+)\)|coordinates?\s*[:\s]+(\d+)[,\s]+(\d+)|x[:\s]+(\d+)[,\s]+y[:\s]+(\d+)', vision_response, re.IGNORECASE)
                        if coord_match:
                            coords = [g for g in coord_match.groups() if g]
                            if len(coords) >= 2:
                                # Coordinates from vision are relative to screenshot, convert to screen coordinates
                                x = window_offset_x + int(coords[0])
                                y = window_offset_y + int(coords[1])
                                print(f"   Clicking at screen coordinates ({x}, {y}) - vision found at relative ({coords[0]}, {coords[1]})")
                                # Ensure browser window is focused before clicking
                                if browser_region:
                                    time.sleep(0.3)  # Give window time to be fully focused
                                pyautogui.click(x, y)
                                return f"Found '{search_text}' using vision model. Clicked at screen coordinates ({x}, {y})."
                        return f"OCR failed. Vision analysis: {vision_response}. Could not extract exact coordinates. Please try: 'click: x,y' with specific coordinates."
                    except Exception as vision_error:
                        return f"OCR failed and vision processing timed out or errored: {str(vision_error)}. Please try again or manually click on '{search_text}'."
            else:
                # Use vision model to find text location
                try:
                    vision_response = process_image_with_vision(
                        screenshot_path, 
                        question=f"Where is the word '{search_text}' located on the screen? Describe its position in detail, including approximate pixel coordinates (x, y) if you can estimate them, or describe its location relative to other visible elements."
                    )
                    
                    # Try to extract coordinates from vision response
                    coord_match = re.search(r'\((\d+),\s*(\d+)\)|coordinates?\s*[:\s]+(\d+)[,\s]+(\d+)|x[:\s]+(\d+)[,\s]+y[:\s]+(\d+)', vision_response, re.IGNORECASE)
                    if coord_match:
                        # Extract coordinates from match
                        coords = [g for g in coord_match.groups() if g]
                        if len(coords) >= 2:
                            # Coordinates from vision are relative to screenshot, convert to screen coordinates
                            x = window_offset_x + int(coords[0])
                            y = window_offset_y + int(coords[1])
                            print(f"   Clicking at screen coordinates ({x}, {y}) - vision found at relative ({coords[0]}, {coords[1]})")
                            # Ensure browser window is focused before clicking
                            if browser_region:
                                time.sleep(0.3)  # Give window time to be fully focused
                            pyautogui.click(x, y)
                            return f"Found '{search_text}' based on vision analysis. Clicked at screen coordinates ({x}, {y})."
                    
                    return f"Vision analysis: {vision_response}. Could not extract exact coordinates. Please try: 'click: x,y' with specific coordinates, or describe the location more clearly."
                except Exception as vision_error:
                    return f"Vision processing timed out or errored: {str(vision_error)}. Please try again or manually click on '{search_text}'. The page may need more time to load - try waiting a few seconds and then asking again."
        
        elif action.startswith("type:"):
            text = action.replace("type:", "").strip()
            pyautogui.write(text, interval=0.1)
            return f"Typed: {text}"
        
        elif action.startswith("click:"):
            coords = action.replace("click:", "").strip()
            x, y = map(int, coords.split(","))
            pyautogui.click(x, y)
            return f"Clicked at ({x}, {y})"
        
        else:
            return f"Unknown action. Supported: 'open browser:', 'screenshot', 'find_text_and_click: <text>', 'type:', 'click:'"
    
    except Exception as e:
        return f"Error controlling computer: {str(e)}"

# Create LangChain tools
tools = [
    Tool(
        name="WebSearch",
        func=web_search_tool,
        description="Search the web for current information, news, or facts. Input should be a simple search query (2-5 words)."
    ),
    Tool(
        name="DocumentQuery",
        func=document_query_tool,
        description="Query documents that the user has uploaded. Input should be a question about the documents."
    ),
    Tool(
        name="ComputerControl",
        func=computer_control_tool,
        description="Control the computer for automation tasks. Use this to: 'open browser: <url>' (opens URL in new tab), 'screenshot' (takes screenshot), 'find_text_and_click: <text>' (takes screenshot, finds text using OCR/vision, and clicks it), 'type: <text>' (types text), 'click: x,y' (clicks coordinates). DO NOT use this tool to provide answers - use Final Answer instead."
    )
]

# ==================== AGENT SETUP ====================

template = """You are JARVIS, a helpful AI assistant with access to tools.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- Use WebSearch for current info/news (keep queries SHORT: 2-5 words)
- Use DocumentQuery for uploaded documents
- Use ComputerControl ONLY for actual automation tasks (opening browser, taking screenshots, typing into apps, clicking)
- For ComputerControl browser actions: Use 'open browser: <url>' format. For YouTube searches, use 'open browser: youtube.com and search for "query"'
- For finding and clicking text on screen: Use 'find_text_and_click: <text>' - this takes a screenshot, finds the text using OCR/vision, and clicks it automatically. IMPORTANT: If you just opened a browser, wait a moment (2-3 seconds) or take a screenshot first to ensure the page has loaded before trying to find text
- NEVER use ComputerControl to provide answers or responses - always use Final Answer for that
- If you don't need a tool, go straight to Final Answer
- CRITICAL: When WebSearch returns results, look for "EXTRACTED PRICE" or price information in the page content
- If you see "EXTRACTED PRICE" in the search results, use that exact value in your answer
- If price information is in the page content, extract and use the specific price value mentioned
- NEVER make up or guess specific numbers, prices, dates, or statistics that are not in the search results
- If search results don't contain the exact price information requested, tell the user you found sources but they should check those sources for the exact current value
- If WebSearch fails completely, provide your best answer based on your knowledge and use Final Answer
- Be helpful and conversational, but always be honest about what information you have vs. what you don't have

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ==================== GRADIO INTERFACE ====================

def chat(input_data, history: List) -> tuple:
    """Handle chat with unified multimodal input (text + images)"""
    try:
        if history is None:
            history = []
        
        # Extract text and files from multimodal input
        if isinstance(input_data, dict):
            message = input_data.get("text", "").strip()
            files = input_data.get("files", [])
            # Get first image if any
            image = files[0] if files else None
        else:
            # Fallback for old format
            message = str(input_data).strip() if input_data else ""
            image = None
        
        # Handle empty message
        if not message:
            message = "What's in this image?" if image else ""
        
        # If image is provided, process it with vision model first
        if image is not None and image != "":
            try:
                # Save uploaded image temporarily
                import tempfile
                
                # Handle Gradio image input (can be file path, dict, or PIL Image)
                if isinstance(image, str):
                    image_path = image
                elif isinstance(image, dict):
                    # Gradio sometimes returns dict with 'path' key
                    image_path = image.get('path', image.get('name', None))
                    if image_path is None:
                        raise ValueError("Could not extract image path from upload")
                else:
                    # If it's a PIL Image or numpy array, save it
                    temp_dir = tempfile.gettempdir()
                    image_path = os.path.join(temp_dir, "jarvis_vision_temp.png")
                    if hasattr(image, 'save'):
                        image.save(image_path)
                    else:
                        raise ValueError("Unsupported image format")
                
                # Process image with vision model
                vision_question = message if message and message.strip() else "What's in this image?"
                vision_response = process_image_with_vision(image_path, question=vision_question)
                
                # Combine image analysis with user message
                if message and message.strip():
                    combined_input = f"User uploaded an image and asked: {message}\n\nImage analysis: {vision_response}\n\nPlease answer the user's question based on the image analysis."
                else:
                    combined_input = f"User uploaded an image. Image analysis: {vision_response}\n\nPlease describe what you see in the image."
                
                response = agent_executor.invoke({"input": combined_input})
                
                # Add to history with image indicator
                user_content = message if message and message.strip() else "📷 [Image uploaded]"
                history.append({"role": "user", "content": user_content})
                history.append({"role": "assistant", "content": response["output"]})
            except Exception as img_error:
                # If image processing fails, fall back to text-only
                print(f"⚠️ Image processing error: {img_error}")
                if message and message.strip():
                    response = agent_executor.invoke({"input": message})
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response["output"]})
                else:
                    history.append({"role": "user", "content": "📷 [Image upload failed]"})
                    history.append({"role": "assistant", "content": f"Error processing image: {str(img_error)}"})
        else:
            # Regular text-only chat
            if not message or not message.strip():
                return history, None  # Don't process empty messages
            
            response = agent_executor.invoke({"input": message})
            
            # NEW Gradio format with dictionaries
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response["output"]})
        
        return history, None  # Clear input
    except Exception as e:
        if history is None:
            history = []
        message = str(input_data.get("text", "")) if isinstance(input_data, dict) else str(input_data) if input_data else ""
        user_content = message if message else "📷 [Image uploaded]"
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, None

def upload_documents(files):
    """Handle document uploads"""
    if not files:
        return "No files uploaded."
    
    file_paths = [file.name for file in files]
    doc_memory.add_documents(file_paths)
    return f"✅ Uploaded and indexed {len(files)} document(s)"

# Create Gradio interface
with gr.Blocks(title="JARVIS - Unified AI Agent") as demo:
    gr.Markdown("# 🤖 JARVIS - Your Unified Local AI Agent")
    gr.Markdown(f"**Model:** {OLLAMA_MODEL} | **Status:** ✅ Connected")
    
    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(
            label="JARVIS Assistant"
        )
        
        # Unified multimodal input (text + images in one field)
        multimodal_input = gr.MultimodalTextbox(
            file_types=["image"],
            show_label=False,
            placeholder="Type your message here... You can paste images from clipboard or drag and drop them.",
        )
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", scale=1)
        
        multimodal_input.submit(chat, inputs=[multimodal_input, chatbot], outputs=[chatbot, multimodal_input])
        clear_btn.click(lambda: ([], None), None, outputs=[chatbot, multimodal_input])
    
    with gr.Tab("📄 Documents"):
        gr.Markdown("## Upload Documents")
        gr.Markdown("Upload PDF, TXT, or DOCX files for RAG.")
        
        file_upload = gr.File(
            label="Choose Files",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx"]
        )
        upload_btn = gr.Button("Process", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)
        
        upload_btn.click(upload_documents, inputs=[file_upload], outputs=[upload_status])
    
    with gr.Tab("ℹ️ Info"):
        gr.Markdown("""
        ## What I Can Do
        
        **💬 Chat Naturally** - I'll decide when to use tools
        
        **👁️ Vision & Images**
        - Paste images directly in the unified input field (from clipboard)
        - Drag and drop images into the input field
        - Ask questions about images: "What's in this image?", "What does this chart show?"
        
        **🔍 Web Search**
        - "What's the weather?"
        - "AI news"
        - "Python tutorials"
        
        **📚 Document Q&A**
        - Upload files in Documents tab
        - "What does my contract say about X?"
        
        **🖥️ Computer Control**
        - "Open browser to google.com"
        - "Open Google Sheets and click on the word Bills"
        
        ## Tips
        - Keep search queries SHORT (2-5 words work best)
        - Responses take 30-60 seconds (local processing)
        - Upload documents once - they persist
        - Vision processing works with images and screenshots
        """)

# ==================== LAUNCH ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting JARVIS...")
    print(f"📍 Model: {OLLAMA_MODEL}")
    print(f"📍 Ollama: {OLLAMA_BASE_URL}")
    print("="*60 + "\n")
    
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
