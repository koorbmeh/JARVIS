"""
JARVIS LLM Setup
Ollama initialization, agent creation, and tool registration
"""

import warnings
import requests
import base64
from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from config import (
    OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL, MAX_ITERATIONS,
    USE_CHAINED_MODELS, TEXT_MODEL, VISION_MODEL
)
from tools import web_search_tool, document_query_tool, file_management_tool, computer_control_tool

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Ollama.*deprecated.*")
warnings.filterwarnings("ignore", message=".*langchain-ollama.*")


def check_ollama_gpu():
    """Check if Ollama is using GPU"""
    try:
        # Check Ollama's ps endpoint for running models
        ps_response = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if ps_response.status_code == 200:
            running = ps_response.json()
            if running.get("models"):
                print(f"🎮 Ollama is running models (using GPU if available)")
                return True
    except:
        pass
    
    # Default: Ollama uses GPU automatically if available
    print("💡 Ollama automatically uses GPU if available. Check 'ollama ps' to verify GPU usage.")
    return None


# Initialize Ollama
print("🧠 Initializing Ollama connection...")
check_ollama_gpu()

# Global vision_llm for chained models (initialized below if needed)
vision_llm = None

# Suppress deprecation warnings during initialization
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Ollama.*deprecated.*")
    
    if USE_CHAINED_MODELS:
        # Chained approach: Fast text model + lightweight vision model
        print(f"⚡ Using chained models: {TEXT_MODEL} (text) + {VISION_MODEL} (vision)")
        llm = ChatOllama(
            model=TEXT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_predict=1024,  # Increased from 256 to allow longer, more detailed responses
            top_p=0.9,
            top_k=40
        )
        # Vision model (used via tool, not directly)
        vision_llm = ChatOllama(
            model=VISION_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2
        )
        print(f"✅ Text model: {TEXT_MODEL} (fast for tools/reasoning)")
        print(f"✅ Vision model: {VISION_MODEL} (on-demand only)")
    else:
        # Default: Integrated VLM
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,  # Reduced further for faster, more deterministic responses
            num_predict=1024,  # Increased from 256 to allow longer, more detailed responses
            top_p=0.9,  # Nucleus sampling for faster inference
            top_k=40,  # Limit vocabulary for faster inference
            # Enable thinking/reflection mode for Qwen3-VL (helps reduce hallucinations)
            # Note: This is model-specific; Qwen3-VL supports thinking tags in prompts
        )
        print(f"✅ Connected to Ollama: {OLLAMA_MODEL} (integrated VLM)")
    
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

print(f"✅ Using embedding model: {OLLAMA_EMBEDDING_MODEL}")
print("💡 To verify GPU usage: Run 'ollama ps' in terminal or check Ollama service logs")


# Vision tool for chained models (only used when USE_CHAINED_MODELS=True)
def vision_analyze_tool(args: str) -> str:
    """Analyze an image using the vision model. Input format: 'image_path, query'"""
    if not USE_CHAINED_MODELS or vision_llm is None:
        return "Vision tool not available (chained models disabled or not initialized)"
    
    try:
        parts = args.split(',', 1)
        if len(parts) != 2:
            return "Error: Input must be 'image_path, query'"
        
        image_path = parts[0].strip()
        query = parts[1].strip() if len(parts) > 1 else "What's in this image?"
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Call vision model
        messages = [HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": query}
        ])]
        
        result = vision_llm.invoke(messages)
        return result.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# Create LangChain tools
tools = [
    Tool(
        name="WebSearch",
        func=web_search_tool,
        description="Search the web for current information using DuckDuckGo. Input: simple search query (2-5 words). Returns: search results with extracted prices (if applicable). ALWAYS cite sources - NEVER fabricate URLs or results. Use this for: current prices, news, weather, recent events, or any time-sensitive information."
    ),
    Tool(
        name="DocumentQuery",
        func=document_query_tool,
        description="Query documents that the user has uploaded to RAG. Input: a question about the documents. Returns: relevant excerpts from uploaded documents. Use this proactively when documents might contain relevant information. If documents don't contain the answer, use WebSearch to find current information."
    ),
    Tool(
        name="FileManagement",
        func=file_management_tool,
        description="Create, read, and manage files in the documents directory. Commands: 'create: <filename> | <content>' (creates file, auto-adds to RAG if .txt/.pdf/.docx), 'read: <filename>' (reads file), 'list' (lists all files), 'add_to_rag: <filename>' (adds existing file to RAG). Use this proactively to save information, create notes, organize data, or manage files as part of completing tasks."
    ),
    Tool(
        name="ComputerControl",
        func=computer_control_tool,
        description="Control the computer for automation tasks using PyAutoGUI. Commands: 'open browser: <url>' (opens URL in new tab), 'screenshot' (takes screenshot), 'find_text_and_click: <text>' (takes screenshot, finds text using OCR/vision, clicks it), 'type: <text>' (types text), 'click: x,y' (clicks coordinates). Use this autonomously to accomplish tasks: open websites, interact with UI, navigate applications, automate workflows. Chain actions: take screenshot → analyze → click → verify → continue. Always verify actions via screenshot OCR when possible. DO NOT use this tool to provide answers - use Final Answer instead."
    )
]

# Add vision tool if using chained models
if USE_CHAINED_MODELS:
    tools.append(
        Tool(
            name="VisionAnalyze",
            func=vision_analyze_tool,
            description="Analyze images/screenshots using vision model. Use ONLY if query involves images, UI, screenshots, or visual content. Input format: 'path/to/image.png, Analyze this screenshot for hotkeys' or 'path/to/image.png, What text is visible?'. For text queries without images, do NOT use this tool."
        )
    )
    print("✅ Vision tool added (chained models mode)")


# Agent prompt template with grounding, reflection, and autonomous action
template = """You are JARVIS, an autonomous AI agent. You can take actions proactively to accomplish goals. ALWAYS ground responses in tools or facts—NO hallucinations.

<thinking>
You have a "thinking" mode: After each tool call, reflect on the observation:
1. Does the result match the query? 
2. Is the information reliable (from tool, not invented)?
3. What's the next logical step to complete the task?
4. Should I take additional actions or proceed to Final Answer?
Use this reflection to self-correct and plan next steps autonomously.
</thinking>

AUTONOMOUS ACTION GUIDELINES:
- **Take initiative**: If a task requires multiple steps, execute them automatically without asking for permission
- **Chain actions**: Use multiple tools in sequence to complete complex tasks (e.g., search → analyze → act)
- **Proactive problem-solving**: If you encounter an issue, try alternative approaches automatically
- **Complete workflows**: Don't stop at partial solutions—finish the entire task
- **Infer intent**: Understand what the user wants to accomplish and take the necessary actions

IMPORTANT: You have voice input/output capabilities! When users ask for voice responses or speak to you, you CAN provide voice responses. The system will automatically convert your text responses to speech when voice mode is enabled. You do NOT need to use any tools for this - just provide your normal text response and the system handles TTS automatically.

You have access to the following tools:

{tools}

Use the following format (grounded in tools):

Question: the input question you must answer
Thought: [Reason step-by-step. What tool fits? Why?]
Action: the action to take, should be one of [{tool_names}]
Action Input: [exact args for the tool - be precise]
Observation: [wait for tool result - NEVER invent this]
Thought: [Reflection: Does observation match query? Is it reliable? Refine if needed.]
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (grounded in tool results)
Final Answer: [the final answer, citing sources from tools - NO hallucinations]

Note: If the question includes "Previous conversation:" context, use that to understand what was discussed earlier. Pay attention to corrections (e.g., if user says "No, that's wrong" or "Check again").

CONVERSATION MEMORY:
- Your conversations are automatically logged and added to RAG for future reference
- If a user asks about something discussed before, use DocumentQuery to search conversation logs
- Example: "What did we talk about yesterday?" → Use DocumentQuery to search conversation_logs folder
- Conversation logs are stored in: documents/conversation_logs/

GROUNDING RULES (CRITICAL - NO HALLUCINATIONS):
- **WebSearch**: Use proactively for any queries needing current information; cite sources verbatim, NO fabricating results or URLs
- **DocumentQuery**: Use when user references documents OR when documents might contain relevant information
- **ComputerControl**: Take autonomous actions to accomplish tasks (open URLs, click buttons, type text, take screenshots). Verify actions via screenshot OCR when possible
- **FileManagement**: Use proactively when file operations would help complete the task (create files, read files, manage documents)
- **VisionAnalyze**: Use automatically when images/screenshots are involved or when visual analysis would help
- If no tool fits: Respond directly, but cite evidence. NEVER invent sites/links/prices

AUTONOMOUS ACTION STRATEGY:
- **For greetings/simple chat (hi, hello, how are you, thanks, ok, yes, no): go STRAIGHT to Final Answer - 1 sentence max**
- **Proactively use WebSearch for:**
  * "spot price", "current price", "today's price", "price of [commodity/stock]"
  * "check internet", "search web", "look up", "find current", "get latest"
  * "what's the price of", "how much is [item]"
  * Any question about current/recent prices, news, weather, or events
  * When you need current information to answer accurately
- **NEVER guess or make up prices - if user asks for a price, you MUST use WebSearch**
- **Use DocumentQuery** when documents might contain relevant information (not just when explicitly mentioned)
- **Use FileManagement** proactively when file operations would help (create notes, save results, organize information)
- **Use ComputerControl** autonomously to accomplish tasks:
  * If user wants to "open a website" → use ComputerControl to open browser
  * If user wants to "find and click something" → take screenshot, analyze, then click
  * If user wants to "type something" → use ComputerControl to type
  * Chain actions: screenshot → analyze → click → verify → continue
- **For complex tasks**: Break them down and execute all steps autonomously
- **For general knowledge questions** (what is X, how does X work, explain X): Use WebSearch if current info needed, otherwise go STRAIGHT to Final Answer
- Keep responses concise but complete (1-2 sentences for simple, 2-4 for complex tasks)

CRITICAL FOR PRICES - WORKFLOW (with reflection):
1. Thought: User wants price. I need WebSearch - cannot guess.
2. Action: WebSearch
3. Action Input: [exact search query]
4. Observation: [WebSearch results - look for "⭐ EXTRACTED PRICE: $XX.XX ⭐"]
5. Thought: [Reflection] Does this match? Is price extracted? If yes, proceed. If no, say "Couldn't find price."
6. Final Answer: Use EXACT price from "EXTRACTED PRICE" - copy verbatim. Format: "The current spot price of [item] is $XX.XX per ounce." (or appropriate unit)
7. NEVER make up, guess, or estimate - ONLY use extracted prices from tool

Example workflow for price query:
Thought: User wants the spot price of silver. I need to use WebSearch - I cannot guess prices.
Action: WebSearch
Action Input: spot price of silver today
Observation: [WebSearch results with "⭐ EXTRACTED PRICE: $58.39 per ounce ⭐"]
Thought: [Reflection] I found the extracted price in the observation. It's reliable (from tool). I should go directly to Final Answer now.
Final Answer: The current spot price of silver is $58.39 per ounce.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


prompt = PromptTemplate.from_template(template)

# Add conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Disable verbose for speed (less logging overhead)
    max_iterations=MAX_ITERATIONS,  # Allow up to 10 iterations for complex questions
    handle_parsing_errors=True,
    return_intermediate_steps=False,  # Don't return intermediate steps to speed up
    max_execution_time=600,  # Timeout after 10 minutes (600s) to allow time for complex reasoning
    memory=memory  # Add conversation memory
)
