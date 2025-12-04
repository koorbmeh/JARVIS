"""
JARVIS Tools
All tool functions for the agent (web search, computer control, file management, document query)
"""

import os
import re
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup
import pyautogui
import webbrowser
from config import DOCUMENTS_DIR, WEB_SEARCH_TIMEOUT, WEB_SEARCH_MAX_CONTENT
from utils import get_browser_window_region, TESSERACT_AVAILABLE
from vision import process_image_with_vision

# Import Tesseract if available
if TESSERACT_AVAILABLE:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        TESSERACT_AVAILABLE = False


def web_search_tool(query: str) -> str:
    """Search the web for current information using DuckDuckGo HTML interface"""
    try:
        print(f"🔍 Searching web for: {query}")
        
        # Check if query is about prices - only attempt price extraction for price-related queries
        price_keywords = ['price', 'cost', 'spot price', 'current price', 'how much', 'cost of', 
                          'silver price', 'gold price', 'commodity price', 'stock price', 
                          'crypto price', 'bitcoin price', 'ethereum price', '$', 'usd']
        is_price_query = any(keyword in query.lower() for keyword in price_keywords)
        
        # Use direct HTTP request to DuckDuckGo HTML interface (works when DDGS package fails)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        
        response = requests.post(url, data=params, headers=headers, timeout=WEB_SEARCH_TIMEOUT)
        
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
        
        # Skip ad/redirect URLs and prioritize reputable financial sites (only for price queries)
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
                    page_response = requests.get(url, headers=headers, timeout=WEB_SEARCH_TIMEOUT, allow_redirects=True)
                    if page_response.status_code == 200:
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                        # Extract text content (remove scripts, styles)
                        for script in page_soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        page_text = page_soup.get_text(separator=' ', strip=True)
                        # Get less content for speed
                        page_text = ' '.join(page_text.split()[:WEB_SEARCH_MAX_CONTENT])
                        result['page_content'] = page_text
                        
                        # Only attempt price extraction if this is a price-related query
                        if is_price_query:
                            # Try to extract price from content - improved patterns for better accuracy
                            # Look for price patterns like $XX.XX per ounce, spot price, etc.
                            price_patterns = [
                                # Pattern 1: $XX.XX per ounce (most common)
                                r'\$[\d,]+\.?\d+\s*(?:per\s+)?(?:ounce|oz|troy\s+ounce|troy\s+oz)',
                                # Pattern 2: Spot price: $XX.XX or Current price: $XX.XX
                                r'(?:spot\s+price|current\s+price|live\s+price|price)[:\s=]+\$?([\d,]+\.?\d+)',
                                # Pattern 3: $XX.XX USD
                                r'\$([\d,]+\.?\d+)\s*(?:USD|US\s+dollar|dollars?)',
                                # Pattern 4: Bid/Ask prices (common on financial sites)
                                r'(?:bid|ask|price)[:\s]+\$?([\d,]+\.?\d+)',
                                # Pattern 5: Just $XX.XX near "silver" or "gold" keywords
                                r'\$([\d,]+\.?\d+)(?=\s*(?:per\s+ounce|oz|USD|silver|gold))',
                            ]
                            extracted_prices = []
                            for pattern in price_patterns:
                                matches = re.findall(pattern, page_text, re.IGNORECASE)
                                if matches:
                                    # Clean up the match (remove commas, ensure it's a valid price)
                                    for match in matches:
                                        if isinstance(match, tuple):
                                            match = match[0] if match[0] else match[1] if len(match) > 1 else str(match)
                                        price_str = str(match).replace(',', '')
                                        try:
                                            price_float = float(price_str)
                                            # Sanity check: silver is typically $15-50/oz, gold $1500-3000/oz
                                            if 10 <= price_float <= 100:  # Likely silver
                                                extracted_prices.append(f"${price_float:.2f} per ounce")
                                            elif 1000 <= price_float <= 5000:  # Likely gold
                                                extracted_prices.append(f"${price_float:.2f} per ounce")
                                        except:
                                            pass
                                    if extracted_prices:
                                        # Use the first valid price found
                                        result['extracted_price'] = extracted_prices[0]
                                        print(f"   ✅ Extracted price: {extracted_prices[0]}")
                                        break
                            
                            # If no price found with patterns, try to find any $XX.XX in the first 500 chars
                            if 'extracted_price' not in result or not result.get('extracted_price'):
                                price_simple = re.search(r'\$([\d,]+\.\d{2})', page_text[:500])
                                if price_simple:
                                    price_val = float(price_simple.group(1).replace(',', ''))
                                    if 10 <= price_val <= 100 or 1000 <= price_val <= 5000:
                                        result['extracted_price'] = f"${price_val:.2f} per ounce (approximate)"
                                        print(f"   ✅ Extracted approximate price: ${price_val:.2f}")
                                # Only log warning if we were actually looking for a price
                                # (removed the warning message to reduce log spam)
                        
                        # Only fetch from one reputable site to avoid rate limiting
                        if is_reputable:
                            break
            except Exception as e:
                print(f"   Could not fetch {result['url']}: {e}")
                result['page_content'] = None
        
        output = f"Web search results for '{query}':\n\n"
        # Check if any result has an extracted price - prioritize it
        has_price = any('extracted_price' in r and r.get('extracted_price') for r in results)
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   {result['body']}\n"
            # Make extracted price VERY prominent
            if 'extracted_price' in result and result['extracted_price']:
                output += f"   ⭐ EXTRACTED PRICE: {result['extracted_price']} ⭐\n"
                output += f"   ⚠️  USE THIS EXACT PRICE - DO NOT GUESS OR MAKE UP A PRICE ⚠️\n"
            if 'page_content' in result and result['page_content']:
                # Include relevant excerpt from page (truncated for speed)
                content = result['page_content'][:WEB_SEARCH_MAX_CONTENT]
                output += f"   Page content: {content}...\n"
            output += f"   URL: {result['url']}\n\n"
        
        # Add summary if price was found
        if has_price:
            price_results = [r for r in results if 'extracted_price' in r and r.get('extracted_price')]
            if price_results:
                output += f"\n📊 PRICE SUMMARY:\n"
                for r in price_results:
                    output += f"   - {r.get('extracted_price')} (from {r.get('title', 'unknown')})\n"
                output += f"\n⚠️  CRITICAL: Use the EXTRACTED PRICE values above. DO NOT make up or guess prices.\n"
        
        return output
    except Exception as e:
        return f"Error searching web: {str(e)}\nTip: Try simpler search terms."


def document_query_tool(query: str) -> str:
    """Query uploaded documents using RAG"""
    # Import here to avoid circular dependency
    from rag import doc_memory
    print(f"📚 Querying documents: {query}")
    return doc_memory.query(query)


def file_management_tool(action: str) -> str:
    """
    Manage files in the documents directory. Supported actions:
    - 'create: <filename> | <content>' - Creates a new file with the specified content
    - 'read: <filename>' - Reads the contents of a file
    - 'list' - Lists all files in the documents directory
    - 'add_to_rag: <filename>' - Adds an existing file to the RAG system for querying
    """
    try:
        print(f"📁 File management: {action}")
        
        # Import here to avoid circular dependency
        from rag import doc_memory
        
        # Ensure documents directory exists
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        
        if action.startswith("create:"):
            # Parse: "create: filename.txt | content here"
            parts = action.replace("create:", "").strip().split("|", 1)
            if len(parts) != 2:
                return "Error: Invalid format. Use 'create: <filename> | <content>'"
            
            filename = parts[0].strip()
            content = parts[1].strip()
            
            # Sanitize filename - prevent directory traversal
            filename = os.path.basename(filename)  # Remove any path components
            
            # Ensure filename has an extension or default to .txt
            if not any(filename.endswith(ext) for ext in ['.txt', '.md', '.py', '.json', '.csv', '.log']):
                filename = filename + '.txt'
            
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Automatically add to RAG if it's a supported format
            supported_extensions = ['.txt', '.pdf', '.docx']
            if any(file_path.endswith(ext) for ext in supported_extensions):
                try:
                    doc_memory.add_documents([file_path])
                    return f"✅ Created file '{filename}' and added it to document memory (RAG). File saved to: {file_path}"
                except Exception as e:
                    return f"✅ Created file '{filename}' but error adding to RAG: {str(e)}. File saved to: {file_path}"
            else:
                return f"✅ Created file '{filename}'. File saved to: {file_path} (Note: This file type is not automatically added to RAG. Supported: .txt, .pdf, .docx)"
        
        elif action.startswith("read:"):
            filename = action.replace("read:", "").strip()
            filename = os.path.basename(filename)  # Sanitize
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            
            if not os.path.exists(file_path):
                return f"Error: File '{filename}' not found in documents directory."
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"Contents of '{filename}':\n\n{content}"
            except Exception as e:
                return f"Error reading file '{filename}': {str(e)}"
        
        elif action == "list":
            try:
                files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))]
                if not files:
                    return "No files found in documents directory."
                
                file_list = "\n".join([f"  - {f}" for f in sorted(files)])
                return f"Files in documents directory ({len(files)} total):\n{file_list}"
            except Exception as e:
                return f"Error listing files: {str(e)}"
        
        elif action.startswith("add_to_rag:"):
            filename = action.replace("add_to_rag:", "").strip()
            filename = os.path.basename(filename)  # Sanitize
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            
            if not os.path.exists(file_path):
                return f"Error: File '{filename}' not found in documents directory."
            
            supported_extensions = ['.txt', '.pdf', '.docx']
            if not any(file_path.endswith(ext) for ext in supported_extensions):
                return f"Error: File type not supported for RAG. Supported formats: .txt, .pdf, .docx"
            
            try:
                doc_memory.add_documents([file_path])
                return f"✅ Added '{filename}' to document memory (RAG). You can now query it using DocumentQuery."
            except Exception as e:
                return f"Error adding '{filename}' to RAG: {str(e)}"
        
        else:
            return "Unknown action. Supported: 'create: <filename> | <content>', 'read: <filename>', 'list', 'add_to_rag: <filename>'"
    
    except Exception as e:
        return f"Error managing files: {str(e)}"


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

