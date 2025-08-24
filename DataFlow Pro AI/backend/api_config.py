"""
API Configuration for DataFlow Pro
Manages connections to multiple backend services
"""

import os
from typing import Optional

class APIConfig:
    """Configuration class for managing API endpoints and keys"""
    
    def __init__(self):
        # Data Science API (original backend)
        self.data_science_api_url = os.getenv("DATA_SCIENCE_API_URL", "http://localhost:8000")
        
        # Report Generator API (new backend)
        self.report_generator_api_url = os.getenv("NEXT_PUBLIC_REPORT_API_URL", "http://localhost:8001")
        
        # Chatbot API
        self.chatbot_api_url = os.getenv("CHATBOT_API_URL", "http://localhost:8002")
        
        # GPT-5 API configuration
        self.gpt5_api_base = os.getenv("GPT5_API_BASE", "https://api.aimlapi.com/v1")
        self.gpt5_api_key = os.getenv("GPT5_API_KEY", "")
        self.gpt5_model = os.getenv("GPT5_MODEL", "openai/gpt-5-chat-latest")
    
    def get_data_science_endpoint(self, path: str) -> str:
        """Get full URL for data science API endpoint"""
        return f"{self.data_science_api_url.rstrip('/')}/{path.lstrip('/')}"
    
    def get_report_generator_endpoint(self, path: str) -> str:
        """Get full URL for report generator API endpoint"""
        return f"{self.report_generator_api_url.rstrip('/')}/{path.lstrip('/')}"
    
    def get_chatbot_endpoint(self, path: str) -> str:
        """Get full URL for chatbot API endpoint"""
        return f"{self.chatbot_api_url.rstrip('/')}/{path.lstrip('/')}"
    
    def validate_config(self) -> dict:
        """Validate API configuration and return status"""
        return {
            "data_science_api": bool(self.data_science_api_url),
            "report_generator_api": bool(self.report_generator_api_url),
            "chatbot_api": bool(self.chatbot_api_url),
            "gpt5_configured": bool(self.gpt5_api_key)
        }

# Global configuration instance
api_config = APIConfig()
