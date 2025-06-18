# src/pipeline/pipeline.py
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Import project-specific modules
from src.config.settings import (
    IMAGE_DIR,
    PDF_DIR,
    OUTPUT_DIR,
    TEMP_IMAGE_OUTPUT_DIR,
    GEMINI_API_KEY
)
from src.generator.qna_generator import QnAGenerator
from src.utils.pdf_to_image_splitter import split_pdf_pages_as_image_pdfs
from src.utils.gemini_api import model # Import the pre-configured Gemini model

class MathQnAPipeline:
    def __init__(
        self,
        input_pdf_folder: Path = PDF_DIR,
        input_image_folder: Path = IMAGE_DIR,
        output_base_dir: Path = OUTPUT_DIR,
        temp_image_output_dir: Path = TEMP_IMAGE_OUTPUT_DIR,
        max_pdf_pages_to_process: int = 7, # Default limit for PDF processing
        process_images_from_folder: bool = True, # Flag to enable/disable processing images from input_image_folder
        process_pdfs_from_folder: bool = False # Flag to enable/disable processing PDFs from input_pdf_folder
    ):
        """
        Initializes the Math Q&A Generation Pipeline.

        Args:
            input_pdf_folder: Directory containing PDF files to process.
            input_image_folder: Directory containing standalone image files to process.
            output_base_dir: Base directory where final Q&A datasets (CSV/JSON) will be saved.
            temp_image_output_dir: Temporary directory for images split from PDFs.
            max_pdf_pages_to_process: Maximum number of pages to process from each PDF.
            process_images_from_folder: If True, images from `input_image_folder` will be processed.
            process_pdfs_from_folder: If True, PDFs from `input_pdf_folder` will be processed.
        """
        self.input_pdf_folder = input_pdf_folder
        self.input_image_folder = input_image_folder
        self.output_base_dir = output_base_dir
        self.temp_image_output_dir = temp_image_output_dir
        self.max_pdf_pages_to_process = max_pdf_pages_to_process
        self.process_images_from_folder = process_images_from_folder
        self.process_pdfs_from_folder = process_pdfs_from_folder

        # Ensure all necessary output/temp directories exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_image_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the QnAGenerator with the globally configured model and output directory
        self.qna_generator = QnAGenerator(model=model, output_dir=self.output_base_dir)

        # NOTE: Gemini API configuration is now handled directly in src/utils/gemini_api.py on import.
        # No explicit configure_gemini() call needed here as `model` is already configured.

    def run(self):
        """
        Executes the complete Math Q&A Generation Pipeline.
        This involves collecting image sources (from direct images and PDFs),
        generating Q&A for each source, and saving the final combined dataset.
        """
        print("\n" + "="*50)
        print("Starting Math Q&A Generation Pipeline...")
        print(f"Using Gemini API Key (first 5 chars): {GEMINI_API_KEY[:5]}*****")
        print("="*50 + "\n")

        all_source_image_paths: List[Path] = []

        # --- Phase 1: Collect Image Paths from Image Directory ---
        if self.process_images_from_folder:
            print("\n--- Phase 1: Collecting Image Paths from dedicated image folder ---")
            supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff'] 
            found_images = []
            for ext in supported_extensions:
                found_images.extend(self.input_image_folder.glob(ext))

            # Filter out any non-files or directories inadvertently picked up
            found_images = [img_path for img_path in found_images if img_path.is_file()]

            if not found_images:
                print(f"[-] No image files found in '{self.input_image_folder}'.")
            else:
                all_source_image_paths.extend(sorted(found_images)) # Sort for consistent order
                print(f"[✓] Found {len(found_images)} image(s) in '{self.input_image_folder}'.")

        # --- Phase 1.5: Split PDFs into Images and Collect Paths ---
        if self.process_pdfs_from_folder:
            print("\n--- Phase 1.5: Splitting PDFs from folder into individual image pages ---")
            pdf_files = sorted(list(self.input_pdf_folder.glob("*.pdf")))

            if not pdf_files:
                print(f"[-] No PDF files found in '{self.input_pdf_folder}'. Skipping PDF processing.")
            else:
                print(f"[✓] Found {len(pdf_files)} PDF(s) in '{self.input_pdf_folder}'.")
                for pdf_file_path in pdf_files:
                    print(f"[*] Processing PDF: {pdf_file_path.name}")
                    # Split PDF pages as images, respecting max_pages limit
                    pdf_page_image_paths = split_pdf_pages_as_image_pdfs(
                        input_pdf_path=pdf_file_path,
                        output_dir=self.temp_image_output_dir,
                        max_pages=self.max_pdf_pages_to_process
                    )
                    all_source_image_paths.extend(pdf_page_image_paths)
                    if pdf_page_image_paths:
                        print(f"[✓] Converted {len(pdf_page_image_paths)} pages from {pdf_file_path.name} to images.")
                    else:
                        print(f"[-] No pages converted from {pdf_file_path.name}.")

        if not all_source_image_paths:
            print("\n[!] No source image files (from image folder or PDFs) to process. Exiting pipeline.")
            return

        print(f"\nTotal unique image sources to process: {len(all_source_image_paths)}")

        extracted_qna_data_combined: List[Dict[str, Any]] = []

        # --- Phase 2: Generating Q&A from All Collected Images ---
        print("\n--- Phase 2: Generating Q&A from Images using Gemini ---")
        for i, image_path in enumerate(all_source_image_paths):
            print(f"\n--- Processing Source Image {i+1}/{len(all_source_image_paths)}: {image_path.name} ---")

            qna_result = self.qna_generator.process_image_for_qna(image_path)

            if qna_result:
                # Add metadata about the original source file
                if self.input_image_folder in image_path.parents:
                    qna_result["original_source_type"] = "image_folder"
                    qna_result["original_source_name"] = image_path.name
                elif self.temp_image_output_dir in image_path.parents:
                    # For PDF-derived images, try to get the original PDF name from the filename stem
                    # e.g., "Abramowitz & Stegun_page_1.png" -> "Abramowitz & Stegun.pdf"
                    original_pdf_name = "_".join(image_path.stem.split('_')[:-2]) + ".pdf"
                    qna_result["original_source_type"] = "pdf_page"
                    qna_result["original_source_name"] = original_pdf_name
                else:
                    qna_result["original_source_type"] = "unknown"
                    qna_result["original_source_name"] = image_path.name # Fallback

                # Generate a unique formula ID based on source and image name
                qna_result["formula_id"] = f"{qna_result['original_source_name'].replace('.', '_')}_{image_path.stem}_id_{i+1}"
                qna_result["image_path_in_outputs"] = str(self.temp_image_output_dir.name) + "/" + image_path.name # Store relative path for outputs

                extracted_qna_data_combined.append(qna_result)
                print(f"  [✓] Successfully processed {image_path.name}. Total Q&A records: {len(extracted_qna_data_combined)}")
            else:
                print(f"  [-] No Q&A data generated for {image_path.name}. Skipping this item.")

        if not extracted_qna_data_combined:
            print("\n[!] No Q&A data generated from any source images after processing. Exiting pipeline.")
            return

        print(f"\n[✓] Finished processing all eligible images. Generated Q&A for {len(extracted_qna_data_combined)} items.")

        # --- Phase 3: Saving Final Results ---
        print("\n--- Phase 3: Saving Final Results ---")

        # Save as JSON
        final_json_filepath = self.output_base_dir / "final_qna_dataset.json"
        try:
            with open(final_json_filepath, 'w', encoding='utf-8') as f:
                json.dump(extracted_qna_data_combined, f, indent=4, ensure_ascii=False)
            print(f"[✓] All Q&A results saved to JSON: {final_json_filepath}")
        except Exception as e:
            print(f"[X] Error saving Q&A data to JSON: {e}")

        # Save as CSV
        final_csv_filepath = self.output_base_dir / "final_qna_dataset.csv"
        try:
            df = pd.DataFrame(extracted_qna_data_combined)
            # Convert list/dict columns to JSON strings for CSV compatibility
            for col in ['critical_expressions', 'topics', 'critical_steps', 'alternate_answers']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)

            df.to_csv(final_csv_filepath, index=False, encoding='utf-8-sig')
            print(f"[✓] Formatted results also saved to CSV: {final_csv_filepath}")
        except Exception as e:
            print(f"[X] Error saving to CSV: {e}. Data might be too complex for direct CSV, consider JSON output only.")


        print("\n" + "="*50)
        print("Pipeline execution complete.")
        print("="*50 + "\n")

        # Overall Metrics for the entire run
        total_prompt_tokens = sum(r.get('total_prompt_tokens_per_question', 0) for r in extracted_qna_data_combined)
        total_candidate_tokens = sum(r.get('total_candidate_tokens_per_question', 0) for r in extracted_qna_data_combined)
        total_api_time = sum(r.get('total_api_time_per_question', 0.0) for r in extracted_qna_data_combined)

        print(f"\nOverall Metrics for this Run:")
        print(f"  Total Prompt Tokens Used: {total_prompt_tokens}")
        print(f"  Total Candidate Tokens Generated: {total_candidate_tokens}")
        print(f"  Total API Time Spent: {total_api_time:.2f} seconds")
        print("\nDon't forget to remove temporary images from 'temp_images' directory if no longer needed.")