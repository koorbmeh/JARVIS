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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# ==================== CONFIGURATION ====================

OLLAMA_MODEL = "qwen3-vl:8b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"

# Global storage for browser drivers (keeps browsers open)
_browser_drivers = []

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

def computer_control_tool(action: str) -> str:
    """
    Control the computer. Supported actions:
    - 'open browser: <url>' - Opens a browser to URL (browser stays open)
    - 'open browser: youtube.com/search?q=<query>' - Opens YouTube and searches
    - 'screenshot' - Takes a screenshot
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
            
            try:
                # Configure Chrome options to use user's regular Chrome profile
                chrome_options = ChromeOptions()
                
                # Use the user's existing Chrome profile so it looks like regular Chrome
                import os
                username = os.getenv('USERNAME') or os.getenv('USER')
                chrome_user_data = f"C:\\Users\\{username}\\AppData\\Local\\Google\\Chrome\\User Data"
                
                # Check if Chrome profile exists
                if os.path.exists(chrome_user_data):
                    chrome_options.add_argument(f"--user-data-dir={chrome_user_data}")
                    chrome_options.add_argument("--profile-directory=Default")
                    print(f"   Using your Chrome profile from: {chrome_user_data}")
                else:
                    # Fallback: create a separate profile but make it look normal
                    print("   Using separate Chrome profile (your regular Chrome may be open)")
                
                # Keep browser open even after script ends
                chrome_options.add_experimental_option("detach", True)
                
                # Don't show "Chrome is being controlled by automated test software" banner
                chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                
                # Make it look more like regular Chrome
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                
                driver = webdriver.Chrome(options=chrome_options)
                
                # Store driver globally to prevent garbage collection
                _browser_drivers.append(driver)
                
                driver.get(url)
                time.sleep(2)  # Give page time to load
                title = driver.title
                
                # Browser will stay open - detach option keeps it alive
                return f"Opened browser to {url}. Page title: {title}. Browser window will remain open - you can close it manually when done."
            except Exception as e:
                return f"Error opening browser: {str(e)}. Please check that Chrome and ChromeDriver are installed correctly."
        
        elif action == "screenshot":
            screenshot = pyautogui.screenshot()
            screenshot_path = "screenshot.png"
            screenshot.save(screenshot_path)
            return f"Screenshot saved to {screenshot_path}"
        
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
            return f"Unknown action. Supported: 'open browser:', 'screenshot', 'type:', 'click:'"
    
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
        description="Control the computer for automation tasks. Use this to: 'open browser: <url>' (opens URL and keeps browser open), 'open browser: youtube.com and search for \"query\"' (opens YouTube and searches), 'screenshot' (takes screenshot), 'type: <text>' (types text), 'click: x,y' (clicks coordinates). DO NOT use this tool to provide answers - use Final Answer instead."
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

def chat(message: str, history: List) -> tuple:
    """Process chat messages through the agent"""
    try:
        if history is None:
            history = []
        
        response = agent_executor.invoke({"input": message})
        
        # NEW Gradio format with dictionaries
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response["output"]})
        
        return history, ""
    except Exception as e:
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""

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
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Ask me anything! I can search the web, query documents, or control your computer.",
                lines=2
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        gr.Examples(
            examples=[
                "Hello JARVIS!",
                "What's the weather in Madison?",
                "Search for AI news",
                "Open browser to google.com"
            ],
            inputs=msg
        )
        
        submit_btn.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(lambda: ([], ""), None, outputs=[chatbot, msg])
    
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
        
        **🔍 Web Search**
        - "What's the weather?"
        - "AI news"
        - "Python tutorials"
        
        **📚 Document Q&A**
        - Upload files in Documents tab
        - "What does my contract say about X?"
        
        **🖥️ Computer Control**
        - "Open browser to google.com"
        - "Take a screenshot"
        
        ## Tips
        - Keep search queries SHORT (2-5 words work best)
        - Responses take 30-60 seconds (local processing)
        - Upload documents once - they persist
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
