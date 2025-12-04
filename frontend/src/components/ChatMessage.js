import React from 'react';
import './ChatMessage.css';

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        {message.image && (
          <div className="message-image">
            <img src={message.image} alt="User upload" />
          </div>
        )}
        {message.content && (
          <div className="message-text">{message.content}</div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;

