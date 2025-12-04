"""
JARVIS Self-Reflection System
Logs errors and slow responses, generates improvement examples for future reference
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict
from config import DOCUMENTS_DIR, ENABLE_SELF_REFLECTION, SLOW_RESPONSE_THRESHOLD

# Self-reflection log directory
REFLECTION_LOG_DIR = os.path.join(DOCUMENTS_DIR, "reflection_logs")
MIN_ERROR_COUNT_FOR_REFLECTION = 3  # Only reflect after N similar errors


def ensure_reflection_directory():
    """Ensure the reflection log directory exists"""
    os.makedirs(REFLECTION_LOG_DIR, exist_ok=True)
    return REFLECTION_LOG_DIR


def log_error_or_slow_response(
    task: str,
    result: str,
    execution_time: float,
    error: Optional[str] = None,
    user_feedback: Optional[str] = None
):
    """
    Log an error or slow response for self-reflection
    
    Args:
        task: The user's query/task
        result: The agent's response
        execution_time: How long it took (seconds)
        error: Error message if any
        user_feedback: User feedback like "that's wrong" or "try again"
    """
    if not ENABLE_SELF_REFLECTION:
        return
    
    try:
        ensure_reflection_directory()
        
        # Determine if this should trigger reflection
        is_slow = execution_time > SLOW_RESPONSE_THRESHOLD
        has_error = error is not None
        has_negative_feedback = user_feedback and any(word in user_feedback.lower() 
                                                      for word in ['wrong', 'incorrect', 'bad', 'no', 'try again', 'fix'])
        
        if not (is_slow or has_error or has_negative_feedback):
            return  # Only log problematic cases
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result,
            "execution_time": execution_time,
            "error": error,
            "user_feedback": user_feedback,
            "issue_type": "error" if has_error else ("slow" if is_slow else "feedback")
        }
        
        # Save to daily log file
        log_file = os.path.join(REFLECTION_LOG_DIR, f"reflections_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"📝 Logged reflection: {log_entry['issue_type']} (time: {execution_time:.1f}s)")
        
        # If we have enough similar errors, generate improvement examples
        if has_error or has_negative_feedback:
            _check_and_generate_improvements(task, error or user_feedback)
            
    except Exception as e:
        print(f"⚠️ Error logging reflection: {e}")


def _check_and_generate_improvements(task: str, issue: str):
    """Check if we have enough similar errors, then generate improvement examples"""
    try:
        ensure_reflection_directory()
        
        # Count similar errors from today
        today_file = os.path.join(REFLECTION_LOG_DIR, f"reflections_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        
        if not os.path.exists(today_file):
            return
        
        similar_count = 0
        with open(today_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("issue_type") in ["error", "feedback"]:
                        similar_count += 1
                except:
                    pass
        
        # Only generate improvements if we have enough examples
        if similar_count >= MIN_ERROR_COUNT_FOR_REFLECTION:
            _generate_improvement_examples(task, issue)
            
    except Exception as e:
        print(f"⚠️ Error checking for improvements: {e}")


def _generate_improvement_examples(task: str, issue: str):
    """Generate improvement examples using the LLM"""
    try:
        from llm_setup import llm
        
        prompt = f"""Task that had an issue: {task}
Issue: {issue}

Generate 3-5 examples of how to handle similar tasks better in the future.
Format as JSON array: [{{"task": "...", "better_approach": "..."}}]

Focus on:
- Using the right tools
- Avoiding hallucinations
- Providing accurate information
- Following user intent correctly"""

        # Generate examples (non-blocking, don't wait too long)
        try:
            response = llm.invoke(prompt)
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                examples = json.loads(json_match.group())
                
                # Save to improvement examples file
                examples_file = os.path.join(REFLECTION_LOG_DIR, "improvement_examples.jsonl")
                for example in examples:
                    with open(examples_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(example) + "\n")
                
                print(f"✅ Generated {len(examples)} improvement examples")
        except Exception as e:
            print(f"⚠️ Could not generate improvement examples: {e}")
            
    except Exception as e:
        print(f"⚠️ Error generating improvements: {e}")


def get_improvement_examples(task: str, limit: int = 3) -> list:
    """Retrieve relevant improvement examples for a task"""
    try:
        examples_file = os.path.join(REFLECTION_LOG_DIR, "improvement_examples.jsonl")
        
        if not os.path.exists(examples_file):
            return []
        
        examples = []
        with open(examples_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    example = json.loads(line)
                    # Simple keyword matching (could be improved with embeddings)
                    if any(word in task.lower() for word in example.get("task", "").lower().split()):
                        examples.append(example)
                        if len(examples) >= limit:
                            break
                except:
                    pass
        
        return examples
    except Exception as e:
        print(f"⚠️ Error retrieving improvement examples: {e}")
        return []

