# main.py
from src.pipeline.pipeline import MathQnAPipeline
from src.config.settings import (
    IMAGE_DIR,
    PDF_DIR,
    OUTPUT_DIR,
    TEMP_IMAGE_OUTPUT_DIR
)

def main():
    """
    Main entry point for the Math Q&A Generation Pipeline.
    Configures and runs the pipeline based on settings.
    """
    print("\n" + "="*60)
    print("                 OMINI PROBLEM GENERATOR")
    print("         Generating Math Q&A from Images and PDFs")
    print("="*60 + "\n")

    # Instantiate the pipeline with configuration from settings.py
    # You can override these defaults here if needed for specific runs
    pipeline = MathQnAPipeline(
        input_pdf_folder=PDF_DIR,
        input_image_folder=IMAGE_DIR,
        output_base_dir=OUTPUT_DIR,
        temp_image_output_dir=TEMP_IMAGE_OUTPUT_DIR,
        max_pdf_pages_to_process=1, # Adjust this to process more pages from each PDF
        process_images_from_folder=False, # Set to True to process images from data/input_images
        process_pdfs_from_folder=True     # Set to True to process PDFs from data/input_pdfs
    )

    # Run the pipeline
    pipeline.run()

    print("\n" + "="*60)
    print("                 Pipeline Finished")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()