"""
JARVIS LLM Setup
Ollama initialization, agent creation, and tool registration
"""

import warnings
import requests
from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from config import OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL, MAX_ITERATIONS
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

# Suppress deprecation warnings during initialization
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Ollama.*deprecated.*")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,  # Reduced further for faster, more deterministic responses
        num_predict=256,  # Reduced from 512 - shorter responses = faster (default is much longer)
        top_p=0.9,  # Nucleus sampling for faster inference
        top_k=40  # Limit vocabulary for faster inference
    )
    
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

print(f"✅ Connected to Ollama: {OLLAMA_MODEL} (LLM)")
print(f"✅ Using embedding model: {OLLAMA_EMBEDDING_MODEL}")
print("💡 To verify GPU usage: Run 'ollama ps' in terminal or check Ollama service logs")


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
        name="FileManagement",
        func=file_management_tool,
        description="Create, read, and manage files in the documents directory. Use 'create: <filename> | <content>' to create a file (automatically added to RAG if .txt/.pdf/.docx), 'read: <filename>' to read a file, 'list' to list all files, or 'add_to_rag: <filename>' to add an existing file to RAG."
    ),
    Tool(
        name="ComputerControl",
        func=computer_control_tool,
        description="Control the computer for automation tasks. Use this to: 'open browser: <url>' (opens URL in new tab), 'screenshot' (takes screenshot), 'find_text_and_click: <text>' (takes screenshot, finds text using OCR/vision, and clicks it), 'type: <text>' (types text), 'click: x,y' (clicks coordinates). DO NOT use this tool to provide answers - use Final Answer instead."
    )
]


# Agent prompt template
template = """You are JARVIS, a helpful AI assistant with access to tools.

IMPORTANT: You have voice input/output capabilities! When users ask for voice responses or speak to you, you CAN provide voice responses. The system will automatically convert your text responses to speech when voice mode is enabled. You do NOT need to use any tools for this - just provide your normal text response and the system handles TTS automatically.

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

Note: If the question includes "Previous conversation:" context, use that to understand what was discussed earlier. Pay attention to corrections (e.g., if user says "No, that's wrong" or "Check again").

Guidelines (SPEED IS CRITICAL - BE SMART):
- **For greetings/simple chat (hi, hello, how are you, thanks, ok, yes, no): go STRAIGHT to Final Answer - 1 sentence max**
- **MANDATORY: ALWAYS use WebSearch when user asks about:**
  * "spot price", "current price", "today's price", "price of [commodity/stock]"
  * "check internet", "search web", "look up", "find current", "get latest"
  * "what's the price of", "how much is [item]"
  * Any question about current/recent prices, news, weather, or events
- **NEVER guess or make up prices - if user asks for a price, you MUST use WebSearch**
- Use DocumentQuery ONLY when user says "my document", "uploaded file", "the file I uploaded"
- Use FileManagement ONLY when user explicitly says "create file", "read file", "list files"
- Use ComputerControl ONLY when user explicitly asks to "open browser", "take screenshot", "click", "type"
- For general knowledge questions (what is X, how does X work, explain X), go STRAIGHT to Final Answer - don't search
- Keep responses SHORT (1-2 sentences for simple questions, 2-3 for complex) - speed matters
- **CRITICAL FOR PRICES - WORKFLOW:**
  1. Use WebSearch tool ONCE to get the price
  2. In the Observation, look for "⭐ EXTRACTED PRICE: $XX.XX ⭐"
  3. IMMEDIATELY go to Final Answer - do NOT use any more tools
  4. Use the EXACT price value from "EXTRACTED PRICE" - copy it verbatim
  5. Format: "The current spot price of [item] is $XX.XX per ounce." (or appropriate unit)
  6. If multiple extracted prices are shown, use the first one or mention the range
  7. NEVER make up, guess, or estimate a price - ONLY use extracted prices
  8. If no "EXTRACTED PRICE" is found, say "I couldn't find the current price. Please check [source URL] directly."
- **Example workflow for price query:**
  Thought: User wants the spot price of silver. I need to use WebSearch.
  Action: WebSearch
  Action Input: spot price of silver today
  Observation: [WebSearch results with "⭐ EXTRACTED PRICE: $58.39 per ounce ⭐"]
  Thought: I found the extracted price. I should go directly to Final Answer now.
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
    max_iterations=MAX_ITERATIONS,  # Reduced from 5 to speed up responses
    handle_parsing_errors=True,
    return_intermediate_steps=False,  # Don't return intermediate steps to speed up
    max_execution_time=45,  # Timeout after 45 seconds (increased to allow web search + processing)
    memory=memory  # Add conversation memory
)

