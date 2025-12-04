"""
JARVIS FastAPI Backend
Main API server for JARVIS AI Agent
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import json
import asyncio
import os
import sys
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)  # Change to project root for relative paths

from config import (
    OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL,
    USE_CHAINED_MODELS, TEXT_MODEL, VISION_MODEL
)
from llm_setup import agent_executor
from vision import process_image_with_vision
from voice import transcribe_audio, text_to_speech, TTS_AVAILABLE
from rag import doc_memory
from conversation_logger import (
    save_conversation_turn, ENABLE_CONVERSATION_LOGGING,
    load_conversation_history_on_startup
)
from self_reflection import log_error_or_slow_response
from debug_logger import log_debug, log_info, log_warning, log_error, get_latest_log

app = FastAPI(title="JARVIS API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: conversation_mode is now passed as a local parameter to avoid race conditions
active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    log_info("="*60)
    log_info("🚀 Starting JARVIS FastAPI Backend...")
    log_info(f"📝 Debug logs: {get_latest_log()}")
    print("\n" + "="*60)
    print("🚀 Starting JARVIS FastAPI Backend...")
    if USE_CHAINED_MODELS:
        print(f"📍 Text Model: {TEXT_MODEL}")
        print(f"📍 Vision Model: {VISION_MODEL}")
    else:
        print(f"📍 LLM Model: {OLLAMA_MODEL}")
    print(f"📍 Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"📍 Ollama: {OLLAMA_BASE_URL}")
    print("="*60 + "\n")
    
    # Load conversation history on startup
    print("\n📚 Loading conversation history...")
    load_conversation_history_on_startup()
    print("")
    
    # Log that backend is ready
    log_info("✅ Backend fully initialized and ready")
    print("✅ Backend fully initialized and ready\n")


@app.get("/api/health")
async def health_check():
    """Health check endpoint - returns healthy only when fully initialized"""
    try:
        # Check if agent_executor is ready (models loaded)
        # This endpoint will only return 200 if the backend is fully ready
        log_debug("Health check requested")
        if agent_executor is None:
            log_warning("Health check failed: agent_executor is None")
            return JSONResponse(
                status_code=503,
                content={"status": "initializing", "message": "Backend is still initializing"}
            )
        log_debug("Health check passed: agent_executor is ready")
        return {
            "status": "healthy",
            "text_model": TEXT_MODEL if USE_CHAINED_MODELS else OLLAMA_MODEL,
            "vision_model": VISION_MODEL if USE_CHAINED_MODELS else None,
            "use_chained_models": USE_CHAINED_MODELS,
            "ready": True
        }
    except Exception as e:
        log_error("Health check failed", error=e)
        return JSONResponse(
            {
                "status": "not_ready",
                "error": str(e),
                "ready": False
            },
            status_code=503
        )

@app.get("/api/debug/logs")
async def get_debug_logs(lines: int = 100):
    """Get the latest debug log entries"""
    from debug_logger import read_latest_log_lines, get_latest_log
    try:
        log_lines = read_latest_log_lines(lines)
        return {
            "log_file": get_latest_log(),
            "lines": log_lines,
            "count": len(log_lines)
        }
    except Exception as e:
        return JSONResponse(
            {"error": f"Error reading logs: {str(e)}"},
            status_code=500
        )

@app.get("/api/debug/errors")
async def get_recent_errors(hours: int = 1, limit: int = 50):
    """Get recent errors and warnings from logs"""
    from debug_logger import read_latest_log_lines, get_latest_log
    import re
    from datetime import datetime, timedelta
    
    try:
        # Read more lines to find errors (errors might be less frequent)
        log_lines = read_latest_log_lines(1000)
        
        # Filter for errors and warnings from the last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        errors = []
        warnings = []
        
        for line in log_lines:
            # Check if line contains error or warning
            if '[ERROR]' in line or '[WARNING]' in line:
                # Try to extract timestamp
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time >= cutoff_time:
                            if '[ERROR]' in line:
                                errors.append({
                                    "timestamp": timestamp_match.group(1),
                                    "message": line.strip()
                                })
                            elif '[WARNING]' in line:
                                warnings.append({
                                    "timestamp": timestamp_match.group(1),
                                    "message": line.strip()
                                })
                    except:
                        # If timestamp parsing fails, include it anyway
                        if '[ERROR]' in line:
                            errors.append({"timestamp": "unknown", "message": line.strip()})
                        elif '[WARNING]' in line:
                            warnings.append({"timestamp": "unknown", "message": line.strip()})
                else:
                    # No timestamp, include if it's recent (in last lines)
                    if len(errors) + len(warnings) < 100:  # Only if we haven't found many yet
                        if '[ERROR]' in line:
                            errors.append({"timestamp": "unknown", "message": line.strip()})
                        elif '[WARNING]' in line:
                            warnings.append({"timestamp": "unknown", "message": line.strip()})
        
        # Limit results
        errors = errors[:limit]
        warnings = warnings[:limit]
        
        return {
            "log_file": get_latest_log(),
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "time_range_hours": hours
        }
    except Exception as e:
        return JSONResponse(
            {"error": f"Error reading logs: {str(e)}"},
            status_code=500
        )

@app.post("/api/shutdown")
async def shutdown():
    """Shutdown endpoint - stops the server and frontend"""
    import sys
    import os
    import subprocess
    import threading
    
    log_info("Shutdown requested via API")
    
    def shutdown_server():
        import time
        time.sleep(1)  # Give time for response to be sent
        
        # Try to stop frontend (React dev server on port 3000)
        try:
            # Find and kill process on port 3000 (Windows)
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['netstat', '-aon'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                node_pids = []
                for line in result.stdout.split('\n'):
                    if ':3000' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            node_pids.append(pid)
                            try:
                                # Kill the Node process and its entire process tree (/T flag)
                                # This should also kill the parent cmd.exe window
                                subprocess.run(['taskkill', '/F', '/T', '/PID', pid], 
                                             capture_output=True, timeout=2)
                                log_info(f"Frontend Node process and tree stopped (PID: {pid})")
                            except:
                                pass
                
                # Kill by window title (closes the terminal window) - try multiple patterns
                for title_pattern in ['JARVIS Frontend*', '*JARVIS Frontend*', 'JARVIS Frontend']:
                    try:
                        subprocess.run(['taskkill', '/F', '/FI', f'WINDOWTITLE eq {title_pattern}'], 
                                     capture_output=True, timeout=2)
                        log_info(f"Frontend terminal window closed (pattern: {title_pattern})")
                    except:
                        pass
                
                # Also kill any remaining node.exe processes related to React
                try:
                    subprocess.run(['taskkill', '/F', '/IM', 'node.exe', '/FI', 'COMMANDLINE eq *react-scripts*'], 
                                 capture_output=True, timeout=2)
                except:
                    pass
        except Exception as e:
            log_warning("Error stopping frontend", error=e)
        
        # Stop backend - kill by port to ensure it's actually stopped
        try:
            # Find and kill process on port 8000 (Windows)
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['netstat', '-aon'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if ':8000' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            try:
                                # Kill the process
                                subprocess.run(['taskkill', '/F', '/PID', pid], 
                                             capture_output=True, timeout=2)
                                log_info("Backend process stopped")
                            except:
                                pass
                # Kill by window title (closes the terminal window)
                try:
                    subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq JARVIS Backend*'], 
                                 capture_output=True, timeout=2)
                except:
                    pass
        except Exception as e:
            log_warning("Error stopping backend by port", error=e)
        
        # Also kill the launch script window if it exists (try multiple title patterns)
        try:
            if os.name == 'nt':  # Windows
                # Try different window title patterns
                for title_pattern in ['launch_jarvis.bat*', '*launch_jarvis*']:
                    try:
                        subprocess.run(['taskkill', '/F', '/FI', f'WINDOWTITLE eq {title_pattern}'], 
                                     capture_output=True, timeout=2)
                    except:
                        pass
        except:
            pass
        
        # Also try direct exit
        try:
            os._exit(0)  # Force exit (more reliable than sys.exit in async context)
        except:
            try:
                sys.exit(0)
            except:
                pass
    
    # Start shutdown in a separate thread
    threading.Thread(target=shutdown_server, daemon=True).start()
    
    return {"status": "shutting_down", "message": "JARVIS is shutting down. The browser window will close automatically."}

@app.get("/api/debug/status")
async def get_debug_status():
    """Get current system status and recent issues"""
    from debug_logger import get_latest_log
    import os
    
    try:
        status = {
            "backend_running": True,
            "log_file": get_latest_log(),
            "log_file_exists": os.path.exists(get_latest_log()) if get_latest_log() else False,
            "recent_errors": []
        }
        
        # Get recent errors (last hour)
        try:
            from debug_logger import read_latest_log_lines
            log_lines = read_latest_log_lines(100)
            error_count = sum(1 for line in log_lines if '[ERROR]' in line)
            warning_count = sum(1 for line in log_lines if '[WARNING]' in line)
            
            status["recent_error_count"] = error_count
            status["recent_warning_count"] = warning_count
            
            # Get last few errors
            errors = [line.strip() for line in log_lines if '[ERROR]' in line][-5:]
            status["recent_errors"] = errors
        except:
            pass
        
        return status
    except Exception as e:
        return JSONResponse(
            {"error": f"Error getting status: {str(e)}"},
            status_code=500
        )


@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
    conversation_mode: bool = Form(False)
):
    """Handle chat requests with text and optional image"""
    start_time = time.time()
    image_path = None  # Initialize at function level for cleanup in exception handler
    
    try:
        # Handle image if provided
        if image:
            try:
                # Check if file was actually uploaded
                filename = getattr(image, 'filename', None)
                if filename:
                    # Read content first before creating file
                    content = await image.read()
                    if content:  # Only create file if we have content
                        import tempfile
                        import uuid
                        temp_dir = tempfile.gettempdir()
                        # Preserve original extension
                        ext = os.path.splitext(filename)[1] or '.png'
                        # Use UUID for unique filename to avoid race conditions with concurrent requests
                        image_path = os.path.join(temp_dir, f"jarvis_vision_{uuid.uuid4().hex}{ext}")
                        with open(image_path, "wb") as f:
                            f.write(content)
                    else:
                        image_path = None  # Empty file - don't create temp file
            except Exception as e:
                print(f"⚠️ Error reading image file: {e}")
                image_path = None  # Treat as no image if there's an error
        
        # Process with agent
        if image_path:
            # Process image with vision model first
            vision_question = message if message.strip() else "What's in this image?"
            vision_response = process_image_with_vision(image_path, question=vision_question)
            
            if message.strip():
                combined_input = f"User uploaded an image and asked: {message}\n\nImage analysis: {vision_response}\n\nPlease answer the user's question based on the image analysis."
            else:
                combined_input = f"User uploaded an image. Image analysis: {vision_response}\n\nPlease describe what you see in the image."
        else:
            combined_input = message
        
        response = agent_executor.invoke({"input": combined_input})
        response_text = response["output"]
        
        # Clean up response - remove timeout/limit messages if we have actual content
        if "Agent stopped due to iteration limit or time limit" in response_text:
            parts = response_text.split("Agent stopped due to iteration limit or time limit")
            if parts[0].strip():
                # We have content before the error, use it
                response_text = parts[0].strip()
            else:
                # No content, replace with a user-friendly message
                response_text = "I apologize, but I wasn't able to complete my response within the time limit. Please try rephrasing your question or breaking it into smaller parts."
        
        execution_time = time.time() - start_time
        log_info("Agent response received", execution_time=execution_time, response_length=len(response_text))
        
        # Log conversation
        if ENABLE_CONVERSATION_LOGGING:
            save_conversation_turn(message, response_text)
        
        # Log slow responses
        log_error_or_slow_response(combined_input, response_text, execution_time)
        
        # Generate TTS if conversation mode is active
        tts_path = None
        if conversation_mode and TTS_AVAILABLE:
            try:
                tts_path = text_to_speech(response_text)
                # Convert to URL path for frontend
                if tts_path and os.path.exists(tts_path):
                    tts_path = f"/api/tts/{os.path.basename(tts_path)}"
            except Exception as e:
                print(f"⚠️ TTS error: {e}")
        
        # Clean up image file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        return JSONResponse({
            "response": response_text,
            "tts_path": tts_path,
            "execution_time": execution_time
        })
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        log_error_or_slow_response(message, error_msg, execution_time, error=str(e))
        
        # Clean up image file on error as well
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        return JSONResponse(
            {"error": error_msg},
            status_code=500
        )


@app.post("/api/voice/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Transcribe audio to text"""
    import tempfile
    import time
    
    request_id = f"transcribe_{int(time.time() * 1000)}"
    log_debug(f"[{request_id}] Transcription request received")
    
    try:
        # Check if audio file was actually uploaded
        log_debug(f"[{request_id}] Checking audio file", filename=audio.filename, has_file=audio.file is not None)
        if not audio.filename and not audio.file:
            log_warning(f"[{request_id}] No audio file provided")
            return JSONResponse(
                {"error": "No audio file provided"},
                status_code=400
            )
        
        temp_dir = tempfile.gettempdir()
        # Preserve original extension or use webm (default from browser)
        filename = audio.filename or 'recording.webm'
        ext = os.path.splitext(filename)[1] or '.webm'
        audio_path = os.path.join(temp_dir, f"jarvis_voice_{int(time.time())}{ext}")
        
        log_debug(f"[{request_id}] Reading audio content", temp_path=audio_path, extension=ext)
        
        # Read and save audio file
        try:
            content = await audio.read()
            log_debug(f"[{request_id}] Audio content read", size=len(content) if content else 0)
        except Exception as e:
            log_error(f"[{request_id}] Error reading audio content", error=e)
            return JSONResponse(
                {"error": f"Error reading audio: {str(e)}"},
                status_code=500
            )
        
        if not content or len(content) == 0:
            log_warning(f"[{request_id}] Empty audio file")
            return JSONResponse(
                {"error": "Empty audio file"},
                status_code=400
            )
        
        try:
            with open(audio_path, "wb") as f:
                f.write(content)
            log_debug(f"[{request_id}] Audio file saved", path=audio_path, size=len(content))
        except Exception as e:
            log_error(f"[{request_id}] Error saving audio file", error=e, path=audio_path)
            return JSONResponse(
                {"error": f"Error saving audio file: {str(e)}"},
                status_code=500
            )
        
        # Verify file was written
        if not os.path.exists(audio_path):
            log_error(f"[{request_id}] Audio file not found after write", path=audio_path)
            return JSONResponse(
                {"error": "Failed to save audio file"},
                status_code=500
            )
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            log_error(f"[{request_id}] Audio file is empty", path=audio_path)
            return JSONResponse(
                {"error": "Failed to save audio file"},
                status_code=500
            )
        
        log_debug(f"[{request_id}] Audio file verified", path=audio_path, size=file_size)
        
        # Transcribe - check if Whisper is available first
        log_debug(f"[{request_id}] Checking Whisper availability")
        from voice import WHISPER_AVAILABLE, FASTER_WHISPER_AVAILABLE, faster_whisper_model
        
        if not WHISPER_AVAILABLE:
            log_error(f"[{request_id}] Whisper not available")
            return JSONResponse(
                {"error": "Whisper not available. Please check installation."},
                status_code=503
            )
        
        log_debug(f"[{request_id}] Whisper available", faster_whisper=FASTER_WHISPER_AVAILABLE, model_loaded=faster_whisper_model is not None)
        
        # Ensure model is initialized (might not be on first request)
        if FASTER_WHISPER_AVAILABLE:
            if faster_whisper_model is None:
                log_info(f"[{request_id}] Initializing Whisper model (first request)")
                from voice import initialize_whisper
                try:
                    init_result = initialize_whisper()
                    log_debug(f"[{request_id}] Whisper initialization result", success=init_result)
                    if not init_result:
                        log_error(f"[{request_id}] Failed to initialize Whisper model")
                        return JSONResponse(
                            {"error": "Could not initialize Whisper model. Please try again."},
                            status_code=503
                        )
                except Exception as e:
                    log_error(f"[{request_id}] Exception during Whisper initialization", error=e)
                    return JSONResponse(
                        {"error": f"Error initializing Whisper: {str(e)}"},
                        status_code=500
                    )
        
        # Transcribe
        log_info(f"[{request_id}] Starting transcription", path=audio_path)
        try:
            transcribed = transcribe_audio(audio_path)
            log_debug(f"[{request_id}] Transcription completed", length=len(transcribed) if transcribed else 0)
        except Exception as e:
            log_error(f"[{request_id}] Exception during transcription", error=e)
            # Clean up temp file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            return JSONResponse(
                {"error": f"Transcription error: {str(e)}"},
                status_code=500
            )
        
        # Check if transcription returned an error message
        if transcribed.startswith("Error:"):
            log_error(f"[{request_id}] Transcription returned error", error_msg=transcribed)
            # Clean up temp file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            return JSONResponse(
                {"error": transcribed},
                status_code=500
            )
        
        log_info(f"[{request_id}] Transcription successful", text_length=len(transcribed))
        
        # Clean up temp file after transcription
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                log_debug(f"[{request_id}] Temp file cleaned up")
        except Exception as e:
            log_warning(f"[{request_id}] Error cleaning up temp file", error=e)
        
        # Ensure transcribed is a string and not empty
        if not transcribed:
            log_error(f"[{request_id}] Transcription returned empty string")
            return JSONResponse(
                {"error": "Transcription returned empty result"},
                status_code=500
            )
        
        if not isinstance(transcribed, str):
            transcribed = str(transcribed)
        
        try:
            response = JSONResponse({"text": transcribed})
            log_debug(f"[{request_id}] Sending response", response_length=len(transcribed))
            return response
        except Exception as e:
            log_error(f"[{request_id}] Error creating JSONResponse", error=e)
            return JSONResponse(
                {"error": f"Error creating response: {str(e)}"},
                status_code=500
            )
        
    except Exception as e:
        log_error(f"[{request_id}] Unhandled exception in transcription endpoint", error=e)
        return JSONResponse(
            {"error": f"Transcription error: {str(e)}"},
            status_code=500
        )


@app.get("/api/tts/{filename}")
async def get_tts(filename: str):
    """Serve TTS audio files"""
    import tempfile
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses"""
    try:
        await websocket.accept()
        log_info("WebSocket connection accepted")
        active_connections.append(websocket)
    except Exception as e:
        log_error("Error accepting WebSocket connection", error=e)
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            image_data = data.get("image", None)  # Base64 encoded image
            conversation_mode = data.get("conversation_mode", False)
            
            log_debug("WebSocket chat request received", message_length=len(message), has_image=image_data is not None)
            
            # Handle image if provided
            image_path = None
            if image_data:
                try:
                    import tempfile
                    import base64
                    import uuid
                    temp_dir = tempfile.gettempdir()
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                    # Use UUID for unique filename to avoid race conditions with concurrent requests
                    image_path = os.path.join(temp_dir, f"jarvis_vision_ws_{uuid.uuid4().hex}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    log_debug("WebSocket image saved", path=image_path)
                except Exception as e:
                    log_error("Error processing WebSocket image", error=e)
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing image: {str(e)}"
                    })
                    continue
            
            # Prepare input
            if image_path:
                vision_question = message if message.strip() else "What's in this image?"
                vision_response = process_image_with_vision(image_path, question=vision_question)
                
                if message.strip():
                    combined_input = f"User uploaded an image and asked: {message}\n\nImage analysis: {vision_response}\n\nPlease answer the user's question based on the image analysis."
                else:
                    combined_input = f"User uploaded an image. Image analysis: {vision_response}\n\nPlease describe what you see in the image."
            else:
                combined_input = message
            
            # Stream response using agent_executor.stream()
            try:
                log_info("Starting WebSocket streaming", input_length=len(combined_input))
                
                # Send initial status
                await websocket.send_json({
                    "type": "status",
                    "content": "thinking"
                })
                
                # Get the full response first, then stream it in chunks
                # This provides a better UX than waiting for the entire response
                log_info("Invoking agent executor for streaming")
                response = agent_executor.invoke({"input": combined_input})
                full_response = response.get("output", "")
                
                # Clean up response - remove timeout/limit messages if we have actual content
                if "Agent stopped due to iteration limit or time limit" in full_response:
                    parts = full_response.split("Agent stopped due to iteration limit or time limit")
                    if parts[0].strip():
                        # We have content before the error, use it
                        full_response = parts[0].strip()
                    else:
                        # No content, replace with a user-friendly message
                        full_response = "I apologize, but I wasn't able to complete my response within the time limit. Please try rephrasing your question or breaking it into smaller parts."
                
                # Stream response in chunks for better UX
                # This simulates real-time streaming even though we have the full response
                chunk_size = 30  # Characters per chunk (smaller = smoother streaming)
                delay_between_chunks = 0.03  # 30ms delay between chunks
                
                log_debug("Streaming response in chunks", total_length=len(full_response), chunk_size=chunk_size)
                
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })
                    await asyncio.sleep(delay_between_chunks)  # Small delay for smooth streaming effect
                
                # Send completion status
                await websocket.send_json({
                    "type": "done",
                    "content": ""
                })
                
                log_info("WebSocket streaming completed", response_length=len(full_response))
                
                # Log conversation
                if ENABLE_CONVERSATION_LOGGING and full_response:
                    save_conversation_turn(message, full_response)
                
                # Generate TTS if conversation mode is active
                if conversation_mode and TTS_AVAILABLE and full_response:
                    try:
                        tts_path = text_to_speech(full_response)
                        if tts_path and os.path.exists(tts_path):
                            tts_url = f"/api/tts/{os.path.basename(tts_path)}"
                            await websocket.send_json({
                                "type": "tts",
                                "content": tts_url
                            })
                    except Exception as e:
                        log_warning("TTS generation failed in WebSocket", error=e)
                
                # Clean up image file
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
                        
            except Exception as e:
                log_error("Error in WebSocket streaming", error=e)
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error: {str(e)}"
                })
                
                # Clean up image file on error as well
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
            
    except WebSocketDisconnect:
        log_info("WebSocket disconnected")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        log_error("WebSocket error", error=e)
        if websocket in active_connections:
            active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

