"""
JARVIS Conversation Logger
Saves chat history to files for RAG reference
"""

import os
from datetime import datetime
from typing import List, Dict
from config import (
    DOCUMENTS_DIR, CHROMA_DB_DIR, ENABLE_CONVERSATION_LOGGING, 
    AUTO_ADD_CONVERSATIONS_TO_RAG, LOAD_CONVERSATION_HISTORY_ON_STARTUP,
    CONVERSATION_HISTORY_DAYS
)

# Conversation log directory (within documents folder)
CONVERSATION_LOG_DIR = os.path.join(DOCUMENTS_DIR, "conversation_logs")


def ensure_log_directory():
    """Ensure the conversation log directory exists"""
    os.makedirs(CONVERSATION_LOG_DIR, exist_ok=True)
    return CONVERSATION_LOG_DIR


def format_conversation_entry(role: str, content: str, timestamp: datetime = None) -> str:
    """Format a single conversation entry"""
    if timestamp is None:
        timestamp = datetime.now()
    
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean content (remove markdown formatting that might interfere)
    clean_content = content.strip()
    
    if role == "user":
        return f"[{time_str}] User: {clean_content}\n"
    elif role == "assistant":
        return f"[{time_str}] JARVIS: {clean_content}\n"
    else:
        return f"[{time_str}] {role}: {clean_content}\n"


def save_conversation_turn(user_message: str, assistant_response: str, session_id: str = None):
    """
    Save a single conversation turn to the log file
    
    Args:
        user_message: The user's message
        assistant_response: JARVIS's response
        session_id: Optional session identifier (defaults to date-based)
    """
    if not ENABLE_CONVERSATION_LOGGING:
        return None
    
    try:
        ensure_log_directory()
        
        # Use date-based session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y-%m-%d")
        
        log_file = os.path.join(CONVERSATION_LOG_DIR, f"conversation_{session_id}.txt")
        
        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(format_conversation_entry("user", user_message))
            f.write(format_conversation_entry("assistant", assistant_response))
            f.write("="*80 + "\n")
        
        print(f"💾 Conversation logged to: {log_file}")
        
        # Optionally add to RAG
        if AUTO_ADD_CONVERSATIONS_TO_RAG:
            add_conversation_to_rag(log_file)
        
        return log_file
    except Exception as e:
        print(f"⚠️ Error logging conversation: {e}")
        return None


def save_full_conversation(history: List[Dict], session_id: str = None):
    """
    Save entire conversation history to a log file
    
    Args:
        history: List of conversation messages (format: [{"role": "user", "content": "..."}, ...])
        session_id: Optional session identifier (defaults to date-based)
    """
    if not ENABLE_CONVERSATION_LOGGING or not history:
        return None
    
    try:
        ensure_log_directory()
        
        # Use date-based session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y-%m-%d")
        
        log_file = os.path.join(CONVERSATION_LOG_DIR, f"conversation_{session_id}.txt")
        
        # Write full conversation
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"JARVIS Conversation Log - Session: {session_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for entry in history:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    role = entry["role"]
                    content = entry["content"]
                    if isinstance(content, str) and content.strip():
                        f.write(format_conversation_entry(role, content))
                        f.write("\n")
        
        print(f"💾 Full conversation logged to: {log_file}")
        
        # Optionally add to RAG
        if AUTO_ADD_CONVERSATIONS_TO_RAG:
            add_conversation_to_rag(log_file)
        
        return log_file
    except Exception as e:
        print(f"⚠️ Error logging conversation: {e}")
        return None


def add_conversation_to_rag(log_file: str):
    """Add a conversation log file to RAG for future reference"""
    try:
        from rag import doc_memory
        
        # Add the log file to RAG
        doc_memory.add_documents([log_file])
        print(f"✅ Conversation log added to RAG: {os.path.basename(log_file)}")
    except Exception as e:
        print(f"⚠️ Error adding conversation to RAG: {e}")


def get_recent_conversations(days: int = 7) -> List[str]:
    """Get list of conversation log files from the last N days"""
    try:
        ensure_log_directory()
        
        if not os.path.exists(CONVERSATION_LOG_DIR):
            return []
        
        files = []
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.timestamp() - (days * 24 * 60 * 60)
        
        for filename in os.listdir(CONVERSATION_LOG_DIR):
            if filename.startswith("conversation_") and filename.endswith(".txt"):
                filepath = os.path.join(CONVERSATION_LOG_DIR, filename)
                if os.path.getmtime(filepath) >= cutoff_date:
                    files.append(filepath)
        
        return sorted(files, reverse=True)  # Most recent first
    except Exception as e:
        print(f"⚠️ Error getting recent conversations: {e}")
        return []


def load_conversation_history_on_startup():
    """Load recent conversation logs into RAG on startup for context"""
    if not LOAD_CONVERSATION_HISTORY_ON_STARTUP or not ENABLE_CONVERSATION_LOGGING:
        return
    
    try:
        ensure_log_directory()
        
        # Get recent conversation files
        recent_logs = get_recent_conversations(days=CONVERSATION_HISTORY_DAYS)
        
        if not recent_logs:
            print("📝 No recent conversation logs found to load")
            return
        
        print(f"📚 Loading {len(recent_logs)} conversation log(s) from the last {CONVERSATION_HISTORY_DAYS} days...")
        
        # Import here to ensure llm_setup is initialized first
        from rag import doc_memory
        
        # Add conversation logs to RAG
        # Note: doc_memory will handle loading existing vectorstore or creating new one
        added_count = 0
        skipped_count = 0
        
        for log_file in recent_logs:
            try:
                # Check if file has content
                if os.path.getsize(log_file) > 0:
                    # Check if this file is already in the vectorstore by trying to add it
                    # (add_documents will handle duplicates gracefully)
                    doc_memory.add_documents([log_file])
                    added_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"⚠️ Error loading {os.path.basename(log_file)}: {e}")
                skipped_count += 1
        
        if added_count > 0:
            print(f"✅ Loaded {added_count} conversation log(s) into RAG for context")
            if skipped_count > 0:
                print(f"   (Skipped {skipped_count} empty or duplicate log(s))")
        else:
            if skipped_count > 0:
                print(f"⚠️ All {skipped_count} conversation log(s) were empty or already loaded")
            else:
                print("⚠️ No conversation logs were successfully loaded")
            
    except Exception as e:
        print(f"⚠️ Error loading conversation history on startup: {e}")
        import traceback
        traceback.print_exc()

