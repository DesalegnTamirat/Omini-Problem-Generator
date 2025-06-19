# src/utils/gemini_api.py
import google.generativeai as genai
from src.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash') # Default model for image + text tasks

print("[+] Gemini model initialized and API configured globally.")