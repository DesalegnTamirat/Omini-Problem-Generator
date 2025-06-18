# src/utils/gemini_api.py
import google.generativeai as genai
from src.config import settings

# Configure the Gemini API with the API key from settings
# This ensures the API key is set globally for genai operations
genai.configure(api_key=settings.GEMINI_API_KEY)

# Initialize the GenerativeModel instance
# We are using 'gemini-pro-vision' for multimodal capabilities (image + text)
# and 'gemini-pro' for text-only tasks.
# We'll expose a general 'model' variable for simplicity,
# and other functions can select specific models as needed.
model = genai.GenerativeModel('gemini-pro-vision') # Default model for image + text tasks

print("[+] Gemini model initialized and API configured globally.")

# You might want to define other models if you explicitly use them later, e.g.:
# text_only_model = genai.GenerativeModel('gemini-pro')