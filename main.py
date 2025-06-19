# main.py
from src.pipeline.pipeline import MathQnAPipeline
from src.config.settings import (
    IMAGE_DIR,
    PDF_DIR,
    OUTPUT_DIR,
    TEMP_IMAGE_OUTPUT_DIR,
    GEMINI_API_KEY # <-- Make sure GEMINI_API_KEY is imported here
)
import google.generativeai as genai # <-- Add this import
import os # <-- Add this import if not already there
from src.utils.gemini_api import model 
def main():
    """
    Main entry point for the Math Q&A Generation Pipeline.
    Configures and runs the pipeline based on settings.
    """
    print("\n" + "="*60)
    print("                 OMINI PROBLEM GENERATOR")
    print("         Generating Math Q&A from Images and PDFs")
    print("="*60 + "\n")

    # --- TEMPORARY API KEY AND MODEL TEST ---
    print("[DEBUG] --- Starting API Key and Model Connectivity Test ---")
    print(f"[DEBUG] Loaded API Key (first 5 chars): {GEMINI_API_KEY[:5]}*****")

    # Re-configure genai here to ensure it's set for this test
    # This duplicates configure from src/utils/gemini_api but ensures it's fresh for this test
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        test_model = genai.GenerativeModel(model.model_name) # Using gemini-pro for a simpler text-only test
        test_response = test_model.generate_content("Hello, Gemini! Are you working?")
        print(f"[DEBUG] Simple API test successful! Response: {test_response.text[:50]}...")
    except Exception as e:
        print(f"[DEBUG] Simple API test FAILED: {e}")
        print("[DEBUG] Please ensure your GEMINI_API_KEY is correct in .env and you have network access.")
        print("[DEBUG] --- API Test Failed, Exiting ---")
        return # Exit if the basic test fails
    print("[DEBUG] --- API Test Passed, Proceeding to Pipeline ---")
    # --- END TEMPORARY API KEY AND MODEL TEST ---

    # Instantiate the pipeline with configuration from settings.py
    # You can override these defaults here if needed for specific runs
    pipeline = MathQnAPipeline(
        input_pdf_folder=PDF_DIR,
        input_image_folder=IMAGE_DIR,
        output_base_dir=OUTPUT_DIR,
        temp_image_output_dir=TEMP_IMAGE_OUTPUT_DIR,
        max_pdf_pages_to_process=1, # Adjust this to process more pages from each PDF
        process_images_from_folder=True, # Set to True to process images from data/input_images
        process_pdfs_from_folder=True     # Set to True to process PDFs from data/input_pdfs
    )

    # Run the pipeline
    pipeline.run()

    print("\n" + "="*60)
    print("                 Pipeline Finished")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()