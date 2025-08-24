import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found. Please set it in .env file.")
    return Groq(api_key=api_key)
