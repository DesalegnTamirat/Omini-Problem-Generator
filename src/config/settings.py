# src/config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Determine the base directory of the project dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- API Keys ---
# Load API key from environment variable
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", None) # Default to None if not found

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or as an environment variable.")

# --- Directory Paths ---
IMAGE_DIR: Path = BASE_DIR / "data" / "input_images"
PDF_DIR: Path = BASE_DIR / "data" / "input_pdfs"
OUTPUT_DIR: Path = BASE_DIR / "output"
TEMP_IMAGE_OUTPUT_DIR: Path = BASE_DIR / "temp_images"

# Ensure necessary directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Settings loaded. Base Directory: {BASE_DIR}")
print(f"Image Input Directory: {IMAGE_DIR}")
print(f"PDF Input Directory: {PDF_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")