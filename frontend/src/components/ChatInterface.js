import React, { useState, useRef, useEffect, useCallback } from 'react';
import './ChatInterface.css';
import MultimodalInput from './MultimodalInput';
import ChatMessage from './ChatMessage';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [conversationMode, setConversationMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useStreaming, setUseStreaming] = useState(false); // Disable streaming by default (WebSocket has issues)
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // WebSocket connection management
  useEffect(() => {
    if (useStreaming) {
      // Connect to WebSocket
      const ws = new WebSocket('ws://127.0.0.1:8000/ws/chat');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'status' && data.content === 'thinking') {
          // Agent is thinking, show processing indicator
          setIsProcessing(true);
        } else if (data.type === 'chunk') {
          // Update streaming message with new chunk
          setMessages(prev => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;
            
            if (lastIndex >= 0 && newMessages[lastIndex].role === 'assistant' && newMessages[lastIndex].streaming) {
              // Append chunk to existing streaming message
              newMessages[lastIndex] = {
                ...newMessages[lastIndex],
                content: newMessages[lastIndex].content + data.content
              };
            } else {
              // Create new streaming message
              newMessages.push({
                role: 'assistant',
                content: data.content,
                streaming: true,
                timestamp: new Date()
              });
            }
            return newMessages;
          });
          scrollToBottom();
        } else if (data.type === 'done') {
          // Streaming complete
          setIsProcessing(false);
          setMessages(prev => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;
            if (lastIndex >= 0 && newMessages[lastIndex].role === 'assistant' && newMessages[lastIndex].streaming) {
              newMessages[lastIndex] = {
                ...newMessages[lastIndex],
                streaming: false
              };
            }
            return newMessages;
          });
        } else if (data.type === 'tts') {
          // Play TTS audio
          if (conversationMode) {
            const ttsUrl = data.content.startsWith('http') 
              ? data.content 
              : `http://127.0.0.1:8000${data.content}`;
            const audio = new Audio(ttsUrl);
            audio.play().catch(e => console.error('TTS playback error:', e));
          }
        } else if (data.type === 'error') {
          setIsProcessing(false);
          const errorMessage = {
            role: 'assistant',
            content: `Error: ${data.content}`,
            timestamp: new Date()
          };
          setMessages(prev => [...prev, errorMessage]);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsProcessing(false);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        // Optionally reconnect after a delay
        setTimeout(() => {
          if (useStreaming && !wsRef.current) {
            // Reconnect logic could go here
          }
        }, 3000);
      };

      return () => {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      };
    }
  }, [useStreaming, conversationMode]);

  const handleSend = useCallback(async (text, imageFile) => {
    if (!text.trim() && !imageFile) {
      // Empty input - trigger voice recording
      // This case is now handled by MultimodalInput's internal logic
      return;
    }

    // Add user message
    const userMessage = {
      role: 'user',
      content: text,
      image: imageFile ? URL.createObjectURL(imageFile) : null,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    // Use WebSocket streaming if available, otherwise fall back to HTTP
    if (useStreaming && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        setIsProcessing(true);
        
        // Convert image to base64 if present
        let imageData = null;
        if (imageFile) {
          const reader = new FileReader();
          imageData = await new Promise((resolve, reject) => {
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(imageFile);
          });
        }

        // Send message via WebSocket
        wsRef.current.send(JSON.stringify({
          message: text || '',
          image: imageData,
          conversation_mode: conversationMode
        }));
      } catch (error) {
        console.error('WebSocket send error:', error);
        // Fall back to HTTP
        await sendViaHTTP(text, imageFile);
      }
    } else {
      // Fall back to HTTP POST
      await sendViaHTTP(text, imageFile);
    }
  }, [conversationMode, useStreaming]);

  const sendViaHTTP = async (text, imageFile) => {
    setIsProcessing(true);
    
    try {
      // Prepare form data
      const formData = new FormData();
      formData.append('message', text || '');
      if (imageFile) {
        formData.append('image', imageFile);
      }
      formData.append('conversation_mode', conversationMode);

      // Send to backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData
      });

      // Handle non-JSON responses (like proxy errors)
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
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}. Backend may not be ready yet.`);
        }
        throw new Error('Invalid response from server. Please try again.');
      }

      if (data.error) {
        throw new Error(data.error);
      }

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Play TTS if available
      if (data.tts_path && conversationMode) {
        // Use full URL if it's a relative path
        const ttsUrl = data.tts_path.startsWith('http') 
          ? data.tts_path 
          : `http://127.0.0.1:8000${data.tts_path}`;
        const audio = new Audio(ttsUrl);
        audio.play().catch(e => console.error('TTS playback error:', e));
      }

    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleShutdown = async () => {
    // Shutdown immediately - no confirmation, no alerts, just close
    try {
      // Send shutdown request and wait a moment for it to process
      await fetch('http://127.0.0.1:8000/api/shutdown', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      // Give backend time to process shutdown
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      // Ignore errors - backend might already be shutting down
    }
    
    // Try to close window (browsers may block this)
    try {
      window.close();
    } catch (e) {
      // If window.close() is blocked, try alternative methods
      try {
        window.location.href = 'about:blank';
        setTimeout(() => window.close(), 100);
      } catch (e2) {
        // Last resort: show message
        alert('JARVIS shutdown initiated. Please close this tab manually.');
      }
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-content">
          <div className="header-left">
            <h1>🤖 JARVIS - Your Unified Local AI Agent</h1>
            <div className="model-info">
              Text Model: qwen2.5:7b | Vision Model: qwen3-vl:4b | Status: ✅ Connected
            </div>
          </div>
          <button className="shutdown-button" onClick={handleShutdown} title="Shutdown JARVIS">
            ⛔ Close JARVIS
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">🤖</div>
            <div className="empty-text">JARVIS Assistant</div>
          </div>
        )}
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} />
        ))}
        {isProcessing && (
          <div className="message assistant processing">
            <div className="processing-indicator">Processing...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <div className="input-options">
          <div className="voice-toggle">
            <label>
              <input
                type="checkbox"
                checked={conversationMode}
                onChange={(e) => setConversationMode(e.target.checked)}
              />
              <span>🎙️ Voice Responses</span>
            </label>
          </div>
          <div className="streaming-toggle">
            <label>
              <input
                type="checkbox"
                checked={useStreaming}
                onChange={(e) => setUseStreaming(e.target.checked)}
              />
              <span>⚡ Streaming</span>
            </label>
          </div>
        </div>
        <MultimodalInput onSend={handleSend} isProcessing={isProcessing} />
      </div>
    </div>
  );
};

export default ChatInterface;

