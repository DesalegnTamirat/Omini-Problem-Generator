from src.config import settings
from src.utils import gemini_api, gemini_api_utils, pdf_to_image_splitter
from src.generator import qna_generator
from src.pipeline import pipeline
from . import main

# Orders to check the files to understand the whole project
first = settings.py
second = gemini_api.py
fifth = pdf_to_image_splitter.py
third = gemini_api_utils.py
fourth = qna_generator.py
sixth = pipeline.py
seventh = main.py