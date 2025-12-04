"""
JARVIS Voice Processing
Whisper STT and TTS functionality
"""

import os
import time
import tempfile
from config import WHISPER_MODEL_SIZE, VOICE_MODEL_DIR, RECORD_SECONDS, USE_FASTER_WHISPER

# Initialize Whisper and TTS
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
TTS_AVAILABLE = False
TTS_METHOD = None
whisper_model = None
faster_whisper_model = None
tts_engine = None
piper_model = None
piper_voice = None

# Try faster-whisper first (much faster than openai-whisper)
if USE_FASTER_WHISPER:
    try:
        from faster_whisper import WhisperModel
        FASTER_WHISPER_AVAILABLE = True
        WHISPER_AVAILABLE = True  # Mark as available
        print("🎤 faster-whisper loaded - Speech-to-Text available (4-5x faster!)")
    except ImportError:
        print("⚠️  faster-whisper not available - install with: pip install faster-whisper")
        print("   Falling back to openai-whisper...")
        try:
            import whisper
            WHISPER_AVAILABLE = True
            print("🎤 openai-whisper loaded - Speech-to-Text available (slower)")
        except ImportError:
            print("⚠️  Whisper not available - install with: pip install openai-whisper or faster-whisper")
else:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("🎤 openai-whisper loaded - Speech-to-Text available")
    except ImportError:
        print("⚠️  Whisper not available - install with: pip install openai-whisper")

# Try pyttsx3 first (simpler, more reliable)
try:
    import pyttsx3
    TTS_AVAILABLE = True
    TTS_METHOD = "pyttsx3"
    print("🔊 pyttsx3 loaded - Text-to-Speech available")
except ImportError:
    # Try Piper as alternative
    try:
        from piper import PiperVoice
        from piper.download import ensure_voice_exists, find_voice
        import json
        TTS_AVAILABLE = True
        TTS_METHOD = "piper"
        print("🔊 Piper loaded - Text-to-Speech available")
    except ImportError:
        TTS_AVAILABLE = False
        TTS_METHOD = None
        print("⚠️  TTS not available - install with: pip install pyttsx3 (recommended) or piper-tts")


def check_gpu_available():
    """Check if GPU/CUDA is available for Whisper"""
    try:
        import torch
        print(f"🔍 Checking GPU availability...")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"   CUDA devices: {device_count}")
            print(f"   Device 0: {device_name}")
            print(f"🎮 GPU detected: {device_name}")
            return "cuda"
        else:
            print("💻 No GPU detected, using CPU")
            print("   💡 To use GPU: Install PyTorch with CUDA support:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return "cpu"
    except ImportError:
        print("💻 PyTorch not available, using CPU")
        print("   💡 Install PyTorch: pip install torch")
        return "cpu"
    except Exception as e:
        print(f"💻 GPU check failed: {e}, using CPU")
        return "cpu"


def initialize_whisper():
    """Initialize Whisper model for speech-to-text"""
    global whisper_model, faster_whisper_model
    if not WHISPER_AVAILABLE:
        return False
    
    try:
        if FASTER_WHISPER_AVAILABLE:
            if faster_whisper_model is None:
                # Check for GPU availability
                device = check_gpu_available()
                device_str = "cuda" if device == "cuda" else "cpu"
                
                print(f"🎤 Loading faster-whisper model ({WHISPER_MODEL_SIZE}) on {device_str.upper()}...")
                faster_whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=device_str,
                    compute_type="float16" if device == "cuda" else "int8"  # Use FP16 on GPU, INT8 on CPU for speed
                )
                
                if device == "cuda":
                    print(f"✅ faster-whisper model loaded on GPU - should be very fast!")
                else:
                    print("✅ faster-whisper model loaded on CPU")
            return True
        else:
            # Fallback to openai-whisper
            if whisper_model is None:
                # Check for GPU availability
                device = check_gpu_available()
                
                print(f"🎤 Loading Whisper model ({WHISPER_MODEL_SIZE}) on {device.upper()}...")
                whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
                
                if device == "cuda":
                    print(f"✅ Whisper model loaded on GPU - should be much faster!")
                else:
                    print("✅ Whisper model loaded on CPU")
            return True
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        # Try CPU fallback if GPU fails
        try:
            print("   Attempting CPU fallback...")
            if FASTER_WHISPER_AVAILABLE:
                faster_whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
                print("✅ faster-whisper model loaded on CPU (fallback)")
            else:
                whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")
                print("✅ Whisper model loaded on CPU (fallback)")
            return True
        except Exception as fallback_error:
            print(f"❌ CPU fallback also failed: {fallback_error}")
            return False


def initialize_tts():
    """Initialize TTS engine (pyttsx3 or Piper)"""
    global tts_engine, piper_model, piper_voice
    if not TTS_AVAILABLE:
        return False
    
    try:
        if TTS_METHOD == "pyttsx3":
            if tts_engine is None:
                tts_engine = pyttsx3.init()
                # Set voice properties
                voices = tts_engine.getProperty('voices')
                if voices:
                    # Try to use a better voice if available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            tts_engine.setProperty('voice', voice.id)
                            break
                tts_engine.setProperty('rate', 150)  # Speed of speech
                tts_engine.setProperty('volume', 0.9)  # Volume level
                print("✅ pyttsx3 TTS initialized")
            return True
        
        elif TTS_METHOD == "piper":
            if piper_model is None:
                os.makedirs(VOICE_MODEL_DIR, exist_ok=True)
                # Download a default voice if not exists
                voice_name = "en_US-lessac-medium"
                voice_path = os.path.join(VOICE_MODEL_DIR, voice_name)
                
                if not os.path.exists(voice_path):
                    print(f"🔊 Downloading Piper voice model ({voice_name})...")
                    ensure_voice_exists(voice_name, [VOICE_MODEL_DIR], [VOICE_MODEL_DIR])
                
                model_path, config_path = find_voice(voice_name, [VOICE_MODEL_DIR])
                
                if model_path and config_path:
                    piper_model = PiperVoice.load(model_path, config_path=config_path)
                    piper_voice = voice_name
                    print(f"✅ Piper voice loaded: {voice_name}")
                    return True
                else:
                    print("❌ Could not find Piper voice model")
                    return False
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        return False
    
    return False


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper (faster-whisper if available, else openai-whisper)"""
    try:
        from debug_logger import log_debug, log_info, log_error
    except ImportError:
        # Fallback if debug_logger not available
        def log_debug(*args, **kwargs): pass
        def log_info(*args, **kwargs): pass
        def log_error(*args, **kwargs): pass
    
    log_debug("transcribe_audio called", audio_path=audio_path)
    
    if not WHISPER_AVAILABLE:
        error_msg = "Error: Whisper not available. Install with: pip install faster-whisper (recommended) or openai-whisper"
        log_error("Whisper not available")
        return error_msg
    
    try:
        if FASTER_WHISPER_AVAILABLE:
            if faster_whisper_model is None:
                log_info("faster_whisper_model is None, initializing...")
                if not initialize_whisper():
                    error_msg = "Error: Could not initialize faster-whisper model"
                    log_error("Failed to initialize faster-whisper model")
                    return error_msg
                log_debug("faster_whisper_model initialized", model_loaded=faster_whisper_model is not None)
        else:
            if whisper_model is None:
                log_info("whisper_model is None, initializing...")
                if not initialize_whisper():
                    error_msg = "Error: Could not initialize Whisper model"
                    log_error("Failed to initialize Whisper model")
                    return error_msg
        
        log_info("Starting transcription", audio_path=audio_path)
        print(f"🎤 Transcribing audio from: {audio_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(audio_path):
            error_msg = f"Error: Audio file not found at: {audio_path}"
            log_error("Audio file not found", audio_path=audio_path)
            return error_msg
        
        # Check file size (should be > 0)
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            error_msg = f"Error: Audio file is empty (0 bytes)"
            log_error("Audio file is empty", audio_path=audio_path)
            return error_msg
        log_debug("Audio file verified", size=file_size, size_kb=file_size/1024)
        print(f"   File size: {file_size / 1024:.1f} KB")
        
        # Get absolute path to avoid path issues
        audio_path = os.path.abspath(audio_path)
        print(f"   Using absolute path: {audio_path}")
        
        # Verify file is readable
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(4)
                # Check if it's a valid WAV file (should start with "RIFF")
                if not header.startswith(b'RIFF'):
                    print(f"⚠️  Warning: File may not be a valid WAV file (header: {header})")
        except Exception as e:
            return f"Error: Cannot read audio file: {str(e)}"
        
        print("   Starting Whisper transcription...")
        print(f"   File verified accessible: {audio_path}")
        
        # Load audio using soundfile (doesn't require ffmpeg) and pass as numpy array
        try:
            import soundfile as sf
            import numpy as np
            
            print("   Loading audio file with soundfile...")
            audio_data, sample_rate = sf.read(audio_path)
            print(f"   ✅ Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("   Converted stereo to mono")
            
            # Normalize to float32 if needed
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
        except ImportError:
            print("   ⚠️  soundfile not available, trying file path (may fail if ffmpeg not installed)")
            audio_data = None
        except Exception as e:
            print(f"   ⚠️  Error loading with soundfile: {e}, falling back to file path")
            audio_data = None
        
        # Transcribe with optimized settings
        try:
            start_time = time.time()
            
            if FASTER_WHISPER_AVAILABLE:
                # Use faster-whisper (much faster!)
                print("   Using faster-whisper (4-5x faster)...")
                segments, info = faster_whisper_model.transcribe(
                    audio_path,
                    language="en",
                    beam_size=1,  # Greedy decoding (fastest)
                    best_of=1,
                    temperature=0,
                    vad_filter=True,  # Voice Activity Detection - skip silence
                    vad_parameters=dict(min_silence_duration_ms=500)  # Skip silence > 500ms
                )
                
                # Extract text from segments
                transcribed_text = " ".join([segment.text for segment in segments])
                result = {"text": transcribed_text}
                
                elapsed = time.time() - start_time
                log_info("Transcription successful", elapsed=elapsed, text_length=len(transcribed_text))
                print(f"   ✅ faster-whisper completed in {elapsed:.1f} seconds (much faster!)")
            else:
                # Use openai-whisper (fallback)
                # Check if model is on GPU for fp16 optimization
                import torch
                model_on_gpu = next(whisper_model.parameters()).is_cuda
                use_fp16 = torch.cuda.is_available() and model_on_gpu
                
                if model_on_gpu:
                    print(f"   ✅ Model confirmed on GPU - using FP16 for speed")
                else:
                    print(f"   ⚠️  Model is on CPU - transcription will be slower")
                
                print("   Calling Whisper transcribe()...")
                if audio_data is not None:
                    # Pass numpy array directly (avoids ffmpeg requirement)
                    result = whisper_model.transcribe(
                        audio_data,
                        language="en",
                        verbose=False,  # Disable verbose for speed
                        fp16=use_fp16,  # Use FP16 on GPU for speed, FP32 on CPU
                        condition_on_previous_text=False,  # Faster for short clips
                        beam_size=1,  # Greedy decoding (faster, slightly less accurate)
                        best_of=1,  # Don't sample multiple times
                        temperature=0,  # Deterministic (faster)
                        compression_ratio_threshold=2.4,  # Default
                        logprob_threshold=-1.0,  # Default
                        no_speech_threshold=0.6  # Default
                    )
                else:
                    # Fallback to file path (requires ffmpeg)
                    result = whisper_model.transcribe(
                        audio_path, 
                        language="en",
                        verbose=False,  # Disable verbose for speed
                        fp16=use_fp16,  # Use FP16 on GPU for speed, FP32 on CPU
                        condition_on_previous_text=False,  # Faster for short clips
                        beam_size=1,  # Greedy decoding (faster)
                        best_of=1,  # Don't sample multiple times
                        temperature=0  # Deterministic (faster)
                    )
                
                elapsed = time.time() - start_time
                print(f"   ✅ Whisper completed in {elapsed:.1f} seconds")
            
        except FileNotFoundError as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ FileNotFoundError during transcription:\n{error_trace}")
            if "ffmpeg" in str(e).lower() or "subprocess" in error_trace.lower():
                return f"Error: ffmpeg not found. Whisper requires ffmpeg to process audio files. Please install ffmpeg:\n\nWindows: Download from https://ffmpeg.org/download.html or use: choco install ffmpeg\n\nAlternatively, ensure soundfile is installed: pip install soundfile"
            return f"Error: Audio file not found during transcription: {str(e)}. File path: {audio_path}"
            
        except KeyboardInterrupt:
            return "Error: Transcription was interrupted"
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ Whisper transcription exception:\n{error_trace}")
            return f"Error during Whisper transcription: {str(e)}"
        
        if not result:
            return "Error: Whisper returned None (no result)"
        
        if not isinstance(result, dict):
            return f"Error: Whisper returned unexpected type: {type(result)}"
        
        if "text" not in result:
            return f"Error: Whisper returned invalid result (no text field). Keys: {list(result.keys())}"
        
        transcribed_text = result["text"].strip()
        
        if not transcribed_text:
            # Check if there are segments that might have text
            if "segments" in result and len(result["segments"]) > 0:
                # Try to extract text from segments
                segment_texts = [seg.get("text", "").strip() for seg in result["segments"]]
                transcribed_text = " ".join([t for t in segment_texts if t])
            
            if not transcribed_text:
                return "Error: Whisper returned empty transcription. The audio may be too quiet, contain no speech, or be too short."
        
        print(f"✅ Transcribed ({len(transcribed_text)} chars): {transcribed_text[:100]}...")
        return transcribed_text
        
    except FileNotFoundError as e:
        return f"Error: Audio file not found: {str(e)}"
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            return f"Error: GPU issue - {error_msg}. Try using a smaller Whisper model or ensure CUDA is properly configured."
        return f"Error transcribing audio (RuntimeError): {error_msg}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_error("Exception in transcribe_audio", error=e, traceback=error_details)
        print(f"❌ Transcription error details:\n{error_details}")
        return f"Error transcribing audio: {str(e)}"


def text_to_speech(text: str, output_path: str = None) -> str:
    """Convert text to speech and return audio file path (absolute path)"""
    if not TTS_AVAILABLE:
        return None
    
    try:
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"jarvis_tts_{int(time.time())}.wav")
        
        # Ensure absolute path
        output_path = os.path.abspath(output_path)
        
        # Skip TTS for very short responses (not worth the delay)
        if len(text.strip()) < 20:
            print(f"⚠️  Skipping TTS for very short response: '{text[:30]}...'")
            return None
        
        # Limit text length to prevent very long TTS generation (300 chars max for speed)
        if len(text) > 300:
            text = text[:300] + "..."
            print(f"⚠️  Text truncated to 300 chars for TTS")
        
        print(f"🔊 Generating speech: {text[:50]}...")
        
        if TTS_METHOD == "pyttsx3":
            if tts_engine is None:
                if not initialize_tts():
                    return None
            
            # Use threading with timeout to prevent hanging
            import threading
            
            def generate_speech():
                try:
                    # pyttsx3 can save directly to file
                    tts_engine.save_to_file(text, output_path)
                    tts_engine.runAndWait()
                except Exception as e:
                    print(f"❌ Error in TTS thread: {e}")
            
            # Start TTS generation in a separate thread with timeout
            tts_thread = threading.Thread(target=generate_speech, daemon=True)
            tts_thread.start()
            tts_thread.join(timeout=15)  # 15 second timeout (increased from 10 for reliability)
            
            if tts_thread.is_alive():
                print(f"⚠️  TTS generation timed out after 15 seconds")
                return None
            
            # Verify file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    print(f"✅ Speech saved to: {output_path} ({file_size} bytes)")
                    return output_path
                else:
                    print(f"⚠️  TTS file created but is empty")
                    return None
            else:
                print(f"⚠️  TTS file was not created")
                return None
        
        elif TTS_METHOD == "piper":
            if piper_model is None:
                if not initialize_tts():
                    return None
            
            # Generate audio with Piper
            audio_data = piper_model.synthesize(text)
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            print(f"✅ Speech saved to: {output_path}")
            return output_path
        
        return None
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_microphones():
    """Check for available microphone devices"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        mic_count = 0
        mic_info = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                mic_count += 1
                mic_info.append(f"  - {info['name']} (Channels: {info['maxInputChannels']})")
        p.terminate()
        if mic_count > 0:
            print(f"🎤 Found {mic_count} microphone(s):")
            for info in mic_info[:5]:  # Show first 5
                print(info)
            return True
        else:
            print("⚠️  No microphones detected. Voice input will not work.")
            return False
    except ImportError:
        print("⚠️  pyaudio not installed - cannot detect microphones. Install with: pip install pyaudio")
        return False
    except Exception as e:
        print(f"⚠️  Error checking microphones: {e}")
        return False


# Initialize models on startup
if WHISPER_AVAILABLE:
    initialize_whisper()
if TTS_AVAILABLE:
    initialize_tts()

# Check for available microphones
check_microphones()

