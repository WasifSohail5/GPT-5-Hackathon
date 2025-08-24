"""
Configuration settings for the Chatbot API backend
"""
import os
from typing import Optional

class ChatbotConfig:
    """Configuration class for Chatbot API settings"""
    
    def __init__(self):
        self.api_key = os.getenv("GPT5_API_KEY", "")
        self.base_url = os.getenv("GPT5_API_BASE", "https://api.aimlapi.com/v1")
        self.model = os.getenv("GPT5_MODEL", "openai/gpt-5-chat-latest")
        self.chatbot_api_url = os.getenv("CHATBOT_API_URL", "http://localhost:8002")
    
    def update_api_key(self, api_key: str):
        """Update the API key"""
        self.api_key = api_key
        os.environ["GPT5_API_KEY"] = api_key
    
    def update_chatbot_url(self, url: str):
        """Update the chatbot API URL"""
        self.chatbot_api_url = url
        os.environ["CHATBOT_API_URL"] = url
    
    def is_configured(self) -> bool:
        """Check if the chatbot is properly configured"""
        return bool(self.api_key and self.chatbot_api_url)

# Global configuration instance
chatbot_config = ChatbotConfig()
