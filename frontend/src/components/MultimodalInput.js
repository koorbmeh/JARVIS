import React, { useState, useRef, useCallback, useEffect } from 'react';
import './MultimodalInput.css';

const MultimodalInput = ({ onSend, isProcessing }) => {
  const [text, setText] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const hasContent = text.trim().length > 0 || imageFile !== null;
  const buttonIcon = isRecording ? '⏹' : (hasContent ? '↑' : '🎤');
  const buttonTitle = isRecording ? 'Stop recording' : (hasContent ? 'Send' : 'Voice input');

  const handleTextChange = (e) => {
    setText(e.target.value);
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Try to use webm, fallback to default
      const options = { mimeType: 'audio/webm' };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'audio/webm;codecs=opus';
      }
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        delete options.mimeType; // Use default
      }
      
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        console.log('onstop fired, audioChunks:', audioChunksRef.current.length);
        setIsRecording(false); // Ensure state is updated
        
        // Stop all tracks first
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        // Wait a brief moment for any final data chunks to be collected
        // This helps ensure all audio data is captured, especially on the first recording
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Check if we have any audio data
        if (audioChunksRef.current.length === 0) {
          console.log('No audio data recorded');
          alert('No audio was recorded. Please try again.');
          return;
        }
        
        // Get the MIME type from MediaRecorder (usually audio/webm or audio/ogg)
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const extension = mimeType.includes('webm') ? 'webm' : (mimeType.includes('ogg') ? 'ogg' : 'wav');
        
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        
        // Validate blob size
        if (audioBlob.size === 0) {
          console.log('Audio blob is empty');
          alert('No audio data captured. Please try again.');
          return;
        }
        
        const audioFile = new File([audioBlob], `recording.${extension}`, { type: mimeType });
        
        console.log('Sending audio for transcription, size:', audioBlob.size, 'bytes');
        
        // Send audio to backend for transcription
        const formData = new FormData();
        formData.append('audio', audioFile);

        try {
          const response = await fetch('/api/voice/transcribe', {
            method: 'POST',
            body: formData
          });

          // Check if response is OK
          if (!response.ok) {
            // Try to get error message, but handle non-JSON responses
            let errorMessage = `Server error: ${response.status}`;
            try {
              const errorData = await response.json();
              errorMessage = errorData.error || errorMessage;
            } catch {
              // If response isn't JSON, try to get text
              try {
                const errorText = await response.text();
                if (errorText) {
                  errorMessage = errorText.substring(0, 100); // Limit length
                }
              } catch {
                // Ignore, use default error message
              }
            }
            throw new Error(errorMessage);
          }

          // Parse JSON response - handle proxy errors that return HTML
          let data;
          try {
            const responseText = await response.text();
            // Check if response looks like HTML (proxy error)
            if (responseText.trim().startsWith('<') || responseText.includes('Proxy error')) {
              throw new Error('Backend not ready. Please wait a moment and try again.');
            }
            data = JSON.parse(responseText);
          } catch (parseError) {
            if (parseError instanceof Error && parseError.message.includes('Backend not ready')) {
              throw parseError;
            }
            console.error('JSON parse error:', parseError);
            throw new Error('Invalid response from server. Backend may not be ready yet. Please try again.');
          }

          if (data.error) {
            throw new Error(data.error);
          }
          
          if (data.text) {
            console.log('Transcription received:', data.text);
            // Set transcribed text in input
            setText(data.text);
            // Auto-submit after transcription
            setTimeout(() => {
              onSend(data.text, null);
              setText('');
            }, 100);
          }
        } catch (error) {
          console.error('Transcription error:', error);
          // Check if it's a connection error
          if (error.message.includes('Proxy error') || error.message.includes('ECONNREFUSED') || error.message.includes('Failed to fetch')) {
            alert('Could not connect to backend. Please make sure the backend is running on port 8000.');
          } else {
            alert(`Error transcribing audio: ${error.message || 'Please try again.'}`);
          }
        }
      };

      mediaRecorder.start(100); // Collect data every 100ms for better responsiveness
      setIsRecording(true);
      console.log('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        alert('Microphone access denied. Please allow microphone access in your browser settings.');
      } else {
        alert('Could not access microphone. Please check permissions.');
      }
    }
  }, [onSend]);

  const stopRecording = useCallback(() => {
    console.log('stopRecording called, mediaRecorder state:', mediaRecorderRef.current?.state);
    
    if (mediaRecorderRef.current) {
      try {
        const state = mediaRecorderRef.current.state;
        if (state === 'recording' || state === 'paused') {
          // Request any remaining data before stopping
          // This ensures we capture all audio chunks, especially important for the first recording
          try {
            mediaRecorderRef.current.requestData();
          } catch (e) {
            console.log('requestData not available or failed:', e);
          }
          
          // Small delay to ensure requestData is processed
          setTimeout(() => {
            try {
              if (mediaRecorderRef.current && (mediaRecorderRef.current.state === 'recording' || mediaRecorderRef.current.state === 'paused')) {
                mediaRecorderRef.current.stop();
                console.log('MediaRecorder stopped');
              }
            } catch (error) {
              console.error('Error stopping recording in timeout:', error);
            }
          }, 50);
        } else {
          console.log('MediaRecorder already stopped or in state:', state);
          setIsRecording(false);
        }
      } catch (error) {
        console.error('Error stopping recording:', error);
        setIsRecording(false);
        // Force cleanup
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
      }
      // Note: Don't set isRecording(false) here - let onstop handler do it
      // This ensures the UI stays in recording state until data is collected
    } else {
      // If no recorder but we think we're recording, just reset state
      console.log('No mediaRecorder found, forcing cleanup');
      setIsRecording(false);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  }, []);

  const handleSubmit = useCallback(() => {
    console.log('handleSubmit called, isProcessing:', isProcessing, 'isRecording:', isRecording, 'hasContent:', hasContent);
    
    if (isProcessing && !isRecording) {
      console.log('Blocked: isProcessing and not recording');
      return;
    }
    
    if (!hasContent) {
      // Empty - trigger voice recording
      if (isRecording) {
        console.log('Stopping recording...');
        stopRecording();
      } else {
        console.log('Starting recording...');
        startRecording();
      }
      return;
    }

    onSend(text, imageFile);
    setText('');
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, imageFile, hasContent, onSend, isProcessing, isRecording, startRecording, stopRecording]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (items) {
      for (let item of items) {
        if (item.type.startsWith('image/')) {
          e.preventDefault();
          const file = item.getAsFile();
          setImageFile(file);
          const reader = new FileReader();
          reader.onload = (e) => setImagePreview(e.target.result);
          reader.readAsDataURL(file);
        }
      }
    }
  };

  return (
    <div className="multimodal-input-container">
      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="Preview" />
          <button
            type="button"
            className="remove-image"
            onClick={handleRemoveImage}
            title="Remove image"
          >
            ×
          </button>
        </div>
      )}
      
      <div className="input-wrapper">
        <div className="input-actions">
          <button
            type="button"
            className="attach-button"
            onClick={() => fileInputRef.current?.click()}
            title="Attach image"
          >
            📎
          </button>
        </div>
        
        <textarea
          ref={textareaRef}
          className="text-input"
          placeholder="Type your message here... You can paste images from clipboard or drag and drop them."
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          rows={1}
          disabled={isProcessing}
        />
        
        <button
          type="button"
          className={`submit-button ${isRecording ? 'recording' : (hasContent ? 'send' : 'mic')}`}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Button clicked, isRecording:', isRecording);
            handleSubmit();
          }}
          disabled={isProcessing && !isRecording}
          title={buttonTitle}
        >
          {buttonIcon}
        </button>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );
};

export default MultimodalInput;

