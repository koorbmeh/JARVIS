"""
JARVIS - Unified Local AI Agent
Main entry point with Gradio interface
"""

import os
import time
import tempfile
from typing import List
import gradio as gr

# Import from modules
from config import OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_DB_DIR, DOCUMENTS_DIR, RECORD_SECONDS
from llm_setup import agent_executor
from vision import process_image_with_vision
from voice import transcribe_audio, text_to_speech, TTS_AVAILABLE
from rag import doc_memory

# Conversation mode state
conversation_mode_active = False


def process_voice_input(audio, history: List) -> tuple:
    """Process voice input: transcribe audio and get response, with TTS if conversation mode is active"""
    global conversation_mode_active
    
    try:
        if history is None:
            history = []
        
        if audio is None:
            return history, None, None
        
        # Extract audio path from various formats (Gradio can return dict, string, or tuple)
        audio_path = None
        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, dict):
            audio_path = audio.get('path', audio.get('name', audio.get('file', None)))
        elif isinstance(audio, tuple):
            audio_path = audio[0] if len(audio) > 0 else None
        elif hasattr(audio, 'name'):
            audio_path = audio.name
        elif hasattr(audio, 'path'):
            audio_path = audio.path
        else:
            audio_path = str(audio)
        
        # Clean up path
        if audio_path:
            audio_path = audio_path.strip().strip('"').strip("'")
        
        print(f"🎤 Audio path received: {audio_path}")
        
        if not audio_path:
            history.append({"role": "user", "content": "🎤 [Voice input]"})
            history.append({"role": "assistant", "content": "Error: No audio file received. Please try recording again."})
            return history, None, None
        
        if not os.path.exists(audio_path):
            history.append({"role": "user", "content": "🎤 [Voice input]"})
            history.append({"role": "assistant", "content": f"Error: Audio file not found at: {audio_path}. Please try recording again."})
            print(f"❌ Audio file not found: {audio_path}")
            return history, None, None
        
        # Copy file to a more permanent location to prevent deletion issues
        safe_audio_path = None
        try:
            import shutil
            safe_audio_path = os.path.join(DOCUMENTS_DIR, f"temp_voice_{int(time.time())}.wav")
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            shutil.copy2(audio_path, safe_audio_path)
            print(f"📋 Copied audio to safe location: {safe_audio_path}")
            audio_path = safe_audio_path
        except Exception as e:
            print(f"⚠️  Could not copy audio file: {e}, using original")
            if not os.path.exists(audio_path):
                history.append({"role": "user", "content": "🎤 [Voice input]"})
                history.append({"role": "assistant", "content": f"Error: Audio file became inaccessible. Please try recording again."})
                return history, None, None
        
        # Small delay to ensure file is fully written
        time.sleep(0.2)
        
        # Transcribe audio to text
        transcribed = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not os.path.exists(audio_path):
                    if attempt < max_retries - 1:
                        print(f"⚠️  File disappeared, retrying... (attempt {attempt + 1})")
                        time.sleep(0.5)
                        continue
                    else:
                        transcribed = f"Error: Audio file became inaccessible during processing"
                        break
                
                try:
                    with open(audio_path, 'rb') as test_file:
                        test_file.read(1)
                except Exception as access_error:
                    if attempt < max_retries - 1:
                        print(f"⚠️  File access error, retrying... (attempt {attempt + 1}): {access_error}")
                        time.sleep(0.5)
                        continue
                    else:
                        transcribed = f"Error: Cannot access audio file: {str(access_error)}"
                        break
                
                transcribed = transcribe_audio(audio_path)
                break
            except FileNotFoundError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  FileNotFoundError, retrying... (attempt {attempt + 1})")
                    time.sleep(0.5)
                    continue
                else:
                    transcribed = f"Error: Audio file not found: {str(e)}"
                    break
            except Exception as e:
                transcribed = f"Error transcribing audio: {str(e)}"
                break
        
        if transcribed is None or transcribed.startswith("Error"):
            history.append({"role": "user", "content": "🎤 [Voice input failed]"})
            history.append({"role": "assistant", "content": transcribed or "Error: Transcription failed"})
            try:
                if safe_audio_path and os.path.exists(safe_audio_path):
                    os.remove(safe_audio_path)
            except Exception as cleanup_error:
                print(f"⚠️  Could not clean up temp file: {cleanup_error}")
            return history, None, None
        
        # Clean up temp file after successful transcription
        try:
            if safe_audio_path and os.path.exists(safe_audio_path):
                os.remove(safe_audio_path)
                print(f"🧹 Cleaned up temp audio file")
        except Exception as cleanup_error:
            print(f"⚠️  Could not clean up temp file: {cleanup_error}")
        
        # Build context from conversation history
        context = ""
        if len(history) > 0:
            recent_history = history[-6:] if len(history) > 6 else history
            context_parts = []
            for msg in recent_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # Ensure content is a string (handle case where it might be a list)
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)
                elif not isinstance(content, str):
                    content = str(content)
                content_clean = content.replace("🎤 ", "").replace("📷 ", "").strip()
                if role == "user":
                    context_parts.append(f"User: {content_clean}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content_clean}")
            if context_parts:
                context = "\n".join(context_parts) + "\n\n"
        
        # Add context to input if we have history
        full_input = transcribed
        if context:
            full_input = f"Previous conversation:\n{context}Current question: {transcribed}"
        
        # Process transcribed text like regular chat
        response = agent_executor.invoke({"input": full_input})
        
        # Clean up response - remove "Agent stopped" message if we have a valid answer
        response_text = response["output"]
        if "Agent stopped due to iteration limit or time limit" in response_text:
            # Check if there's actual content before the error message
            parts = response_text.split("Agent stopped due to iteration limit or time limit")
            if parts[0].strip():
                # Use the content before the error message
                response_text = parts[0].strip()
            else:
                # If no content before error, keep the error but make it less alarming
                response_text = response_text.replace(
                    "Agent stopped due to iteration limit or time limit",
                    "Note: Response may be incomplete due to time constraints."
                )
        
        # Add to history
        history.append({"role": "user", "content": f"🎤 {transcribed}"})
        history.append({"role": "assistant", "content": response_text})
        
        # Generate TTS if conversation mode is active
        tts_audio = None
        print(f"🔍 Voice Input TTS Debug: conversation_mode_active={conversation_mode_active}, TTS_AVAILABLE={TTS_AVAILABLE}")
        if conversation_mode_active and TTS_AVAILABLE:
            print(f"🔊 Generating TTS for voice response...")
            tts_audio = text_to_speech(response_text)
            if tts_audio:
                tts_audio = os.path.abspath(tts_audio)
                if os.path.exists(tts_audio):
                    file_size = os.path.getsize(tts_audio)
                    print(f"🔊 TTS audio ready: {tts_audio} ({file_size} bytes)")
                else:
                    print(f"❌ TTS file not found: {tts_audio}")
                    tts_audio = None
        else:
            print(f"⚠️ TTS not generated: conversation_mode_active={conversation_mode_active}, TTS_AVAILABLE={TTS_AVAILABLE}")
        
        return history, None, tts_audio
    except Exception as e:
        if history is None:
            history = []
        history.append({"role": "user", "content": "🎤 [Voice input]"})
        history.append({"role": "assistant", "content": f"Error processing voice: {str(e)}"})
        return history, None, None


def chat(input_data, history: List) -> tuple:
    """Handle chat with unified multimodal input (text + images)"""
    global conversation_mode_active
    
    try:
        if history is None:
            history = []
        
        # Extract text and files from multimodal input
        if isinstance(input_data, dict):
            message = input_data.get("text", "").strip()
            files = input_data.get("files", [])
            image = files[0] if files else None
        else:
            message = str(input_data).strip() if input_data else ""
            image = None
        
        # Handle empty message
        if not message:
            message = "What's in this image?" if image else ""
        
        # If image is provided, process it with vision model first
        if image is not None and image != "":
            try:
                import tempfile
                
                # Handle Gradio image input
                if isinstance(image, str):
                    image_path = image
                elif isinstance(image, dict):
                    image_path = image.get('path', image.get('name', None))
                    if image_path is None:
                        raise ValueError("Could not extract image path from upload")
                else:
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
                
                # Clean up response - remove "Agent stopped" message if we have a valid answer
                response_text = response["output"]
                if "Agent stopped due to iteration limit or time limit" in response_text:
                    # Check if there's actual content before the error message
                    parts = response_text.split("Agent stopped due to iteration limit or time limit")
                    if parts[0].strip():
                        # Use the content before the error message
                        response_text = parts[0].strip()
                    else:
                        # If no content before error, keep the error but make it less alarming
                        response_text = response_text.replace(
                            "Agent stopped due to iteration limit or time limit",
                            "Note: Response may be incomplete due to time constraints."
                        )
                
                # Add to history
                user_content = message if message and message.strip() else "📷 [Image uploaded]"
                history.append({"role": "user", "content": user_content})
                history.append({"role": "assistant", "content": response_text})
                
                # Generate TTS if voice responses are enabled
                tts_audio = None
                if conversation_mode_active and TTS_AVAILABLE:
                    tts_audio = text_to_speech(response_text)
                    if tts_audio:
                        print(f"🔊 TTS audio ready: {tts_audio}")
                
                return history, None, tts_audio
            except Exception as img_error:
                print(f"⚠️ Image processing error: {img_error}")
                if message and message.strip():
                    response = agent_executor.invoke({"input": message})
                    
                    # Clean up response - remove "Agent stopped" message if we have a valid answer
                    response_text = response["output"]
                    if "Agent stopped due to iteration limit or time limit" in response_text:
                        # Check if there's actual content before the error message
                        parts = response_text.split("Agent stopped due to iteration limit or time limit")
                        if parts[0].strip():
                            # Use the content before the error message
                            response_text = parts[0].strip()
                        else:
                            # If no content before error, keep the error but make it less alarming
                            response_text = response_text.replace(
                                "Agent stopped due to iteration limit or time limit",
                                "Note: Response may be incomplete due to time constraints."
                            )
                    
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response_text})
                    
                    tts_audio = None
                    if conversation_mode_active and TTS_AVAILABLE:
                        tts_audio = text_to_speech(response_text)
                        if tts_audio:
                            print(f"🔊 TTS audio ready: {tts_audio}")
                    
                    return history, None, tts_audio
                else:
                    history.append({"role": "user", "content": "📷 [Image upload failed]"})
                    history.append({"role": "assistant", "content": f"Error processing image: {str(img_error)}"})
                    return history, None, None
        else:
            # Regular text-only chat
            if not message or not message.strip():
                return history, None, None
            
            # Fast path for simple greetings (bypass agent for speed)
            message_lower = message.lower().strip()
            simple_greetings = ["hi", "hello", "hey", "hi jarvis", "hello jarvis", "hey jarvis", "thanks", "thank you", "ok", "okay", "yes", "no"]
            if message_lower in simple_greetings:
                # Quick responses without agent overhead
                greetings_responses = {
                    "hi": "Hello! How can I help you?",
                    "hello": "Hi there! What can I do for you?",
                    "hey": "Hey! What's up?",
                    "hi jarvis": "Hello! Ready to assist.",
                    "hello jarvis": "Hi! How can I help?",
                    "hey jarvis": "Hey! What do you need?",
                    "thanks": "You're welcome!",
                    "thank you": "You're welcome!",
                    "ok": "Got it!",
                    "okay": "Got it!",
                    "yes": "Understood.",
                    "no": "Understood."
                }
                quick_response = greetings_responses.get(message_lower, "Hello! How can I help?")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": quick_response})
                
                # Generate TTS if needed
                tts_audio = None
                if conversation_mode_active and TTS_AVAILABLE:
                    tts_audio = text_to_speech(quick_response)
                    if tts_audio:
                        tts_audio = os.path.abspath(tts_audio)
                
                return history, None, tts_audio
            
            # Build context from conversation history for agent
            context = ""
            if len(history) > 0:
                # Include last 3 exchanges for context (6 messages: 3 user + 3 assistant)
                recent_history = history[-6:] if len(history) > 6 else history
                context_parts = []
                for msg in recent_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # Ensure content is a string (handle case where it might be a list)
                    if isinstance(content, list):
                        content = " ".join(str(item) for item in content)
                    elif not isinstance(content, str):
                        content = str(content)
                    # Remove emoji prefixes for cleaner context
                    content_clean = content.replace("🎤 ", "").replace("📷 ", "").strip()
                    if role == "user":
                        context_parts.append(f"User: {content_clean}")
                    elif role == "assistant":
                        context_parts.append(f"Assistant: {content_clean}")
                if context_parts:
                    context = "\n".join(context_parts) + "\n\n"
            
            # Add context to input if we have history
            full_input = message
            if context:
                full_input = f"Previous conversation:\n{context}Current question: {message}"
            
            response = agent_executor.invoke({"input": full_input})
            
            # Clean up response - remove "Agent stopped" message if we have a valid answer
            response_text = response["output"]
            if "Agent stopped due to iteration limit or time limit" in response_text:
                # Check if there's actual content before the error message
                parts = response_text.split("Agent stopped due to iteration limit or time limit")
                if parts[0].strip():
                    # Use the content before the error message
                    response_text = parts[0].strip()
                else:
                    # If no content before error, keep the error but make it less alarming
                    response_text = response_text.replace(
                        "Agent stopped due to iteration limit or time limit",
                        "Note: Response may be incomplete due to time constraints."
                    )
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})
        
        # Generate TTS if voice responses are enabled
        tts_audio = None
        print(f"🔍 TTS Debug: conversation_mode_active={conversation_mode_active}, TTS_AVAILABLE={TTS_AVAILABLE}")
        if conversation_mode_active and TTS_AVAILABLE:
            print(f"🔊 Generating TTS for response...")
            tts_audio = text_to_speech(response_text)
            if tts_audio:
                tts_audio = os.path.abspath(tts_audio)
                if os.path.exists(tts_audio):
                    file_size = os.path.getsize(tts_audio)
                    print(f"🔊 TTS audio ready: {tts_audio} ({file_size} bytes)")
                else:
                    print(f"❌ TTS file not found: {tts_audio}")
                    tts_audio = None
        else:
            print(f"⚠️ TTS not generated: conversation_mode_active={conversation_mode_active}, TTS_AVAILABLE={TTS_AVAILABLE}")
        
        return history, None, tts_audio
    except Exception as e:
        if history is None:
            history = []
        message = str(input_data.get("text", "")) if isinstance(input_data, dict) else str(input_data) if input_data else ""
        user_content = message if message else "📷 [Image uploaded]"
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, None, None


def upload_documents(files):
    """Handle document uploads"""
    if not files:
        return "No files uploaded."
    
    file_paths = [file.name for file in files]
    doc_memory.add_documents(file_paths)
    return f"✅ Uploaded and indexed {len(files)} document(s)"


def list_documents():
    """List all files in the documents directory with their status"""
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))]
        
        if not files:
            return "No files found in documents directory."
        
        output = f"**Files in Documents Directory** ({len(files)} total):\n\n"
        output += f"**Location:** `{os.path.abspath(DOCUMENTS_DIR)}`\n\n"
        
        # Group files by type
        supported_files = [f for f in files if any(f.endswith(ext) for ext in ['.txt', '.pdf', '.docx'])]
        other_files = [f for f in files if f not in supported_files]
        
        if supported_files:
            output += "**Supported Files (can be added to RAG):**\n"
            for f in sorted(supported_files):
                file_path = os.path.join(DOCUMENTS_DIR, f)
                file_size = os.path.getsize(file_path)
                size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                output += f"  • `{f}` ({size_str})\n"
            output += "\n"
        
        if other_files:
            output += "**Other Files:**\n"
            for f in sorted(other_files):
                file_path = os.path.join(DOCUMENTS_DIR, f)
                file_size = os.path.getsize(file_path)
                size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                output += f"  • `{f}` ({size_str})\n"
        
        if doc_memory.documents_loaded:
            output += f"\n✅ **RAG Status:** Documents are indexed and ready to query."
        else:
            output += f"\n⚠️ **RAG Status:** No documents indexed yet. Upload files and click 'Process' to add them to RAG."
        
        return output
    except Exception as e:
        return f"Error listing documents: {str(e)}"


def upload_and_refresh(files):
    """Upload documents and return both status and updated file list"""
    status = upload_documents(files)
    file_list = list_documents()
    return status, file_list


# Create Gradio interface
with gr.Blocks(title="JARVIS - Unified AI Agent") as demo:
    gr.Markdown("# 🤖 JARVIS - Your Unified Local AI Agent")
    gr.Markdown(f"**Model:** {OLLAMA_MODEL} | **Status:** ✅ Connected")
    
    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(
            label="JARVIS Assistant",
            height=500
        )
        
        # Conversation mode state (hidden)
        conversation_mode = gr.State(value=False)
        
        # Unified multimodal input (text + images)
        multimodal_input = gr.MultimodalTextbox(
            file_types=["image"],
            show_label=False,
            placeholder="Type your message here... You can paste images from clipboard or drag and drop them.",
        )
        
        # Simple bottom row: mic button and send button
        with gr.Row():
            mic_btn = gr.Button("🎤", variant="secondary", size="sm", scale=0, min_width=50)
            recording_status = gr.State(value=False)
            send_btn = gr.Button("Send", variant="primary", size="lg", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", size="sm", scale=0, min_width=70)
        
        # Audio output for auto-play TTS
        tts_output = gr.Audio(
            type="filepath", 
            visible=True,
            autoplay=True,
            label="🔊 Voice Response"
        )
        
        # Conversation mode toggle
        conversation_toggle = gr.Checkbox(
            label="🎙️ Voice Responses",
            value=False
        )
        
        # Event handlers
        def record_audio_direct(history, is_conv_mode):
            """Record audio directly using pyaudio when mic button clicked"""
            global conversation_mode_active
            conversation_mode_active = is_conv_mode
            
            if history is None:
                history = []
            
            try:
                import pyaudio
                import wave
                
                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 16000
                
                output_path = os.path.join(tempfile.gettempdir(), f"jarvis_voice_{int(time.time())}.wav")
                
                print(f"🎤 Starting recording for {RECORD_SECONDS} seconds (reduced for speed)...")
                
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                # Save recording
                wf = wave.open(output_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                time.sleep(0.1)
                
                if not os.path.exists(output_path):
                    raise Exception(f"Recording file was not created: {output_path}")
                
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    raise Exception(f"Recording file is empty: {output_path}")
                
                print(f"✅ Recording saved to: {output_path} ({len(frames) * CHUNK / RATE:.1f} seconds, {file_size / 1024:.1f} KB)")
                
                history.append({"role": "assistant", "content": "🎤 Processing audio... (this may take 10-30 seconds)"})
                
                result = process_voice_input(output_path, history)
                
                if len(history) > 0 and history[-1].get("content") == "🎤 Processing audio... (this may take 10-30 seconds)":
                    history.pop()
                
                return result
                
            except ImportError:
                if history is None:
                    history = []
                history.append({"role": "user", "content": "🎤 [Voice input]"})
                history.append({"role": "assistant", "content": "Error: pyaudio not installed. Install with: pip install pyaudio"})
                return history, None, None
            except Exception as e:
                if history is None:
                    history = []
                history.append({"role": "user", "content": "🎤 [Voice input]"})
                history.append({"role": "assistant", "content": f"Error recording audio: {str(e)}"})
                print(f"❌ Recording error: {e}")
                return history, None, None
        
        def process_text_input(input_data, history, is_conv_mode):
            """Process text/image input"""
            global conversation_mode_active
            conversation_mode_active = is_conv_mode
            print(f"🔍 process_text_input: is_conv_mode={is_conv_mode}, setting conversation_mode_active={conversation_mode_active}")
            
            if input_data:
                return chat(input_data, history)
            return history, None, None
        
        # Mic button starts recording immediately
        mic_btn.click(
            record_audio_direct,
            inputs=[chatbot, conversation_mode],
            outputs=[chatbot, multimodal_input, tts_output]
        )
        
        # Send button for text/image
        send_btn.click(
            process_text_input,
            inputs=[multimodal_input, chatbot, conversation_mode],
            outputs=[chatbot, multimodal_input, tts_output]
        )
        
        # Enter key also sends
        multimodal_input.submit(
            process_text_input,
            inputs=[multimodal_input, chatbot, conversation_mode],
            outputs=[chatbot, multimodal_input, tts_output]
        )
        
        # Conversation mode toggle
        def update_conversation_mode(checked):
            global conversation_mode_active
            conversation_mode_active = checked
            print(f"🔍 Checkbox changed: conversation_mode_active set to {checked}")
            return checked
        
        conversation_toggle.change(
            update_conversation_mode,
            inputs=[conversation_toggle],
            outputs=[conversation_mode]
        )
        
        clear_btn.click(
            lambda: ([], False),
            None,
            outputs=[chatbot, conversation_mode]
        )
    
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
        
        gr.Markdown("---")
        gr.Markdown("## Existing Documents")
        gr.Markdown("View all files in your documents directory.")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
            file_list = gr.Markdown(label="Files", value=list_documents())
        
        refresh_btn.click(list_documents, None, outputs=[file_list])
        upload_btn.click(upload_and_refresh, inputs=[file_upload], outputs=[upload_status, file_list])
    
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
        
        **📁 File Management**
        - "Create a file called notes.txt with the content..."
        - "Read the file example.txt"
        - "List all my files"
        - Created files are automatically added to RAG (if .txt, .pdf, or .docx)
        
        **🖥️ Computer Control**
        - "Open browser to google.com"
        - "Open Google Sheets and click on the word Bills"
        
        **🎤 Voice Input & Output**
        - **Voice Input Mode**: Click the microphone button to speak your message (STT only)
        - **Conversation Mode**: Toggle conversation mode for voice responses (STT + TTS)
        - Speak naturally - JARVIS uses Whisper for accurate transcription
        - Responses can be spoken back using text-to-speech
        
        ## Tips
        - Keep search queries SHORT (2-5 words work best)
        - Responses take 30-60 seconds (local processing)
        - Upload documents once - they persist
        - Vision processing works with images and screenshots
        """)


# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting JARVIS...")
    print(f"📍 LLM Model: {OLLAMA_MODEL}")
    print(f"📍 Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"📍 Ollama: {OLLAMA_BASE_URL}")
    print("="*60 + "\n")
    
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

