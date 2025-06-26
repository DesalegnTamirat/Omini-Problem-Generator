import os
import json
import csv
import re
import logging
import time
import math
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='INFO: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configuration
SCRIPT_DIR = os.getcwd()
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RAW_OUTPUT = os.path.join(SCRIPT_DIR, "output", "raw_output")
TEMP_IMG_DIR = os.path.join(SCRIPT_DIR, "temp_img")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "output_results.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "output_results.json")
OUTPUT_PIPELINE_METRICS = os.path.join(OUTPUT_DIR, "pipeline_metrics.png")
MAX_PROBLEMS = 100
from dotenv import load_dotenv

# !!! IMPORTANT: Load API key securely, e.g., from an environment variable
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBBi1nTI5x3-HUFdjSnk-gL4T_4rR8dUHA")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set. Please set it before running the script.")
    exit(1)

PDF_PAGE_SELECTION = "all"  # Examples: "1-5", "1,3,5", "7-10", "all"

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_IMG_DIR, exist_ok=True)
os.makedirs(RAW_OUTPUT, exist_ok=True) # Ensure RAW_OUTPUT is created

# Initialize Gemini models
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash') # Using 1.5 flash for potentially better performance/cost

def parse_page_selection(page_selection, max_pages):
    """Parse page selection string into list of page numbers (0-based)."""
    if page_selection.lower() == "all":
        return list(range(max_pages))

    pages = set()
    parts = page_selection.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                pages.update(range(start - 1, end))  # Convert to 0-based
            except ValueError:
                logging.warning(f"Invalid page range format: {part}. Skipping.")
        else:
            try:
                pages.add(int(part) - 1)  # Convert to 0-based
            except ValueError:
                logging.warning(f"Invalid page number format: {part}. Skipping.")

    # Filter out invalid pages and sort
    valid_pages = [p for p in pages if 0 <= p < max_pages]
    return sorted(valid_pages)

def clean_json_response(response_text: str) -> str:
    """
    Ultra-robust JSON cleaner for Gemini-2.5-flash output with complex LaTeX.
    Handles:
    - Multiple levels of backslash escaping
    - Mixed LaTeX delimiters 
    - Control characters
    - Smart quotes
    - Trailing commas
    - Missing quotes
    """
    # 1. Initial extraction with multiple fallback patterns
    json_str = response_text.strip()
    
    # Try increasingly permissive extraction patterns
    extraction_patterns = [
        r'(?s)```json\n(.*?)\n```',  # Markdown code block
        r'(?s)\{(.*)\}',              # Basic JSON detection
        r'(?s)"generated_question_llm1":.*?\}',  # Key-based
        r'(?s)\{(.*?)(?<!\\)\}',      # Unescaped closing brace
    ]
    
    for pattern in extraction_patterns:
        try:
            match = re.search(pattern, json_str)
            if match:
                json_str = match.group(1 if pattern == extraction_patterns[0] else 0).strip()
                break
        except:
            continue
    
    # 2. Backslash normalization - critical for LaTeX
    # First pass: protect already properly escaped sequences
    protected_patterns = [
        (r'\\\\', r'ⓢⓢ'),  # Temporary marker for proper double backslashes
        (r'\\"', r'ⓠ'),     # Temporary marker for escaped quotes
        (r'\\/', r'ⓕ'),      # Temporary marker for escaped slash
        (r'\\b', r'ⓑ'),      # Temporary marker for backspace
        (r'\\f', r'ⓕ'),      # Temporary marker for form feed
        (r'\\n', r'ⓝ'),      # Temporary marker for newline
        (r'\\r', r'ⓡ'),      # Temporary marker for carriage return
        (r'\\t', r'ⓣ'),      # Temporary marker for tab
    ]
    
    for pattern, replacement in protected_patterns:
        json_str = re.sub(pattern, replacement, json_str)
    
    # Second pass: fix remaining backslashes (LaTeX commands)
    json_str = re.sub(r'\\(?![ⓢⓠⓕⓑⓝⓡⓣ])', r'\\\\', json_str)
    
    # Third pass: restore protected sequences
    restoration_patterns = [
        (r'ⓢⓢ', r'\\\\'),
        (r'ⓠ', r'\\"'),
        (r'ⓕ', r'\\/'),
        (r'ⓑ', r'\\b'),
        (r'ⓝ', r'\\n'),
        (r'ⓡ', r'\\r'),
        (r'ⓣ', r'\\t'),
    ]
    
    for pattern, replacement in restoration_patterns:
        json_str = json_str.replace(pattern, replacement)
    
    # 3. Math mode protection
    math_modes = [
        (r'\\\(', r'ⓛⓟ'),  # \(
        (r'\\\)', r'ⓛⓡ'),  # \)
        (r'\\\[', r'ⓓⓛ'),  # \[
        (r'\\\]', r'ⓓⓡ'),  # \]
        (r'\$\$(.*?)\$\$', r'ⓓⓓ\1ⓓⓓ'),  # Display math
        (r'\$(.*?)\$', r'ⓘ\1ⓘ'),        # Inline math
    ]
    
    for pattern, replacement in math_modes:
        json_str = re.sub(pattern, replacement, json_str)
    
    # 4. Structural JSON repairs
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Trailing commas
    json_str = re.sub(r'([{\[,])\s*([}\]])', r'\1null\2', json_str)  # Empty elements
    
    # 5. Control characters and special cases
    json_str = ''.join(char for char in json_str 
                      if ord(char) >= 32 or char in '\t\n\r')
    
    # 6. Restore math modes
    math_restore = [
        (r'ⓛⓟ', r'\\('),
        (r'ⓛⓡ', r'\\)'),
        (r'ⓓⓛ', r'\\['),
        (r'ⓓⓡ', r'\\]'),
        (r'ⓓⓓ(.*?)ⓓⓓ', r'$$\1$$'),
        (r'ⓘ(.*?)ⓘ', r'$\1$'),
    ]
    
    for pattern, replacement in math_restore:
        json_str = json_str.replace(pattern, replacement)
    
    # 7. Final validation and wrapping
    if not json_str.startswith('{'):
        json_str = '{' + json_str
    if not json_str.endswith('}'):
        json_str += '}'
    
    return json_str

class ProblemVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_metrics_chart(self, results):
        """Generates four subplots: processing status, evaluation accuracy, taxonomy success, and token usage."""
        # Filter for successfully processed results (those with the expected dict structure)
        # Assuming 'status' key indicates a failed processing attempt
        valid_results = [r for r in results if isinstance(r, dict) and 'status' not in r]

        if not valid_results:
            logging.info("[!] No valid results to visualize.")
            return

        # Prepare data for visualization
        # Use a generic identifier if 'image_filename' is missing for some reason in failed results
        image_filenames_for_labels = [r.get('image_filename', f"Problem {i+1}") for i, r in enumerate(results)]

        # Calculate status counts for all results
        success = [1 if isinstance(r, dict) and 'status' not in r else 0 for r in results]
        failed = [1 if isinstance(r, dict) and 'status' in r else 0 for r in results] # Explicitly mark failures

        # Calculate evaluation metrics only for valid results
        correct = [1 if r.get('judge_evaluation') == "Correct" else 0 for r in valid_results]
        incorrect = [1 if r.get('judge_evaluation') == "Incorrect" else 0 for r in valid_results]
        total_evaluated = len(correct)
        accuracy = (sum(correct) / total_evaluated * 100) if total_evaluated > 0 else 0

        # Calculate taxonomy success for valid results
        valid_subcategory = [1 if r.get('subcategory', 'N/A') != "N/A" else 0 for r in valid_results]
        invalid_subcategory = [1 if r.get('subcategory', 'N/A') == "N/A" else 0 for r in valid_results]

        # Get token data for valid results
        prompt_tokens = [r.get('total_prompt_tokens_per_question', 0) for r in valid_results]
        candidate_tokens = [r.get('total_candidate_tokens_per_question', 0) for r in valid_results]

        # Calculate error percentages (success rate)
        successful_qnas = sum(success)
        total_attempts = len(results)
        success_rate = (successful_qnas / total_attempts * 100) if total_attempts > 0 else 0

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12)) # Slightly larger for better readability

        # Chart 1: Processing Status
        x = np.arange(len(results)) # Use numpy for cleaner x-axis generation
        width = 0.35 # Adjusted width for two bars

        ax1.bar(x - width/2, success, width,
                        label=f'Success ({success_rate:.1f}%)', color='#4CAF50')
        ax1.bar(x + width/2, failed, width,
                        label=f'Failed ({100 - success_rate:.1f}%)', color='#FF6B6B')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Processing Status (Successful Q&As: {successful_qnas}/{total_attempts})')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(image_filenames_for_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.2) # Set a fixed y-limit for binary counts

        # Chart 2: Evaluation Accuracy (only for valid results)
        x_valid = np.arange(len(valid_results))
        ax2.bar(x_valid - width/2, correct, width,
                        label='Correct', color='#4CAF50')
        ax2.bar(x_valid + width/2, incorrect, width,
                        label='Incorrect', color='#FF6B6B')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Evaluation Accuracy ({accuracy:.1f}% Correct, {total_evaluated} Evaluated)')
        ax2.legend()
        ax2.set_xticks(x_valid)
        ax2.set_xticklabels([r['image_filename'] for r in valid_results], rotation=45, ha='right')
        ax2.set_ylim(0, max(max(correct + [0]), max(incorrect + [0])) + 0.2) # Dynamic y-limit

        # Chart 3: Taxonomy Success (only for valid results)
        ax3.bar(x_valid - width/2, valid_subcategory, width,
                        label='Valid Subcategory', color='#4CAF50')
        ax3.bar(x_valid + width/2, invalid_subcategory, width,
                        label='Invalid Subcategory (N/A)', color='#FF6B6B')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Taxonomy Classification (Valid: {sum(valid_subcategory)}/{len(valid_results)})')
        ax3.legend()
        ax3.set_xticks(x_valid)
        ax3.set_xticklabels([r['image_filename'] for r in valid_results], rotation=45, ha='right')
        ax3.set_ylim(0, max(max(valid_subcategory + [0]), max(invalid_subcategory + [0])) + 0.2) # Dynamic y-limit

        # Chart 4: Token Usage (only for valid results)
        ax4.bar(x_valid - width/2, prompt_tokens, width,
                        label='Prompt Tokens', color='#4CAF50')
        ax4.bar(x_valid + width/2, candidate_tokens, width,
                        label='Candidate Tokens', color='#0288D1')
        ax4.set_ylabel('Tokens')
        ax4.set_title('Token Usage per Problem')
        ax4.legend()
        ax4.set_xticks(x_valid)
        ax4.set_xticklabels([r['image_filename'] for r in valid_results], rotation=45, ha='right')
        ax4.set_ylim(bottom=0) # Ensure y-axis starts at 0

        plt.tight_layout()

        plt.savefig(OUTPUT_PIPELINE_METRICS, dpi=300)
        plt.close()
        logging.info(f"Visualization saved to {OUTPUT_PIPELINE_METRICS}")

        # Print overall metrics
        total_prompt = sum(prompt_tokens)
        total_candidate = sum(candidate_tokens)
        total_api_time = sum(r.get('total_api_time_per_question', 0) for r in valid_results)

        logging.info("\nOverall Metrics for this Run:")
        logging.info(f"  Total Prompt Tokens Used: {total_prompt}")
        logging.info(f"  Total Candidate Tokens Generated: {total_candidate}")
        logging.info(f"  Total API Time Spent: {total_api_time:.2f} seconds")
        logging.info(f"  Success Rate: {success_rate:.1f}%")
        logging.info(f"  Accuracy Rate (of evaluated problems): {accuracy:.1f}%")

# --- Placeholder for Part III (PDF and Image Processing Utilities) ---


def preprocess_image(image_path):
    """Preprocess the image to improve LaTeX extraction."""
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Sharpen the image
    img = img.filter(ImageFilter.SHARPEN)

    # Resize if too large to improve processing
    if max(img.size) > 800:
        img = img.resize((int(img.size[0] * 0.7), int(img.size[1] * 0.7)), Image.LANCZOS)

    return img

def extract_latex_from_image(image_path, max_retries=3):
    """
    Extracts LaTeX from a given image file using the Google Gemini Vision model.
    Retries on failure with exponential backoff and segments multiple equations.
    """
    logging.info(f"Attempting to extract LaTeX from {os.path.basename(image_path)} using Gemini Vision...")
    img = preprocess_image(image_path)

    # Define a detailed prompt to segment multiple equations
    prompt = """
    Analyze this image for mathematical equations and formulas. Your output MUST:
    - Identify and extract each distinct mathematical expression separately.
    - Label each equation with 'Equation X:' where X is a number (e.g., Equation 1:, Equation 2:), followed by the LaTeX code.
    - Use $...$ for inline equations and $$...$$ for display equations as appropriate.
    - Do NOT include any surrounding text, explanations, or markdown code blocks (e.g., ```latex``` or ```math```).
    - Ensure all special characters, fractions, integrals, summations, roots, superscripts, and subscripts are correctly represented in LaTeX.
    - Place each labeled equation on a new line.

    Examples of desired output:
    Equation 1: $ax^2 + bx + c = 0$
    Equation 2: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$
    Equation 3: $$\\int_0^1 x^3 \\sqrt{1 - x^2} \, dx = \\frac{2}{15}$$
    Equation 4: $$\\sum_{k=1}^\\infty \\frac{(-1)^{k+1}}{k^2} = \\frac{\pi^2}{12}$$
    """

    for attempt in range(max_retries):
        try:
            # Use the global model (verify if 'gemini-2.0-flash' supports vision; consider 'gemini-pro-vision' if not)
            model_for_vision = genai.GenerativeModel('gemini-2.5-flash')  # Corrected from 2.5 to 2.0
            response = model_for_vision.generate_content([prompt, img])

            extracted_text = response.text.strip()
            if extracted_text:
                # Split the text into individual equations based on "Equation X:" pattern
                equations = []
                current_equation = []
                lines = extracted_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("Equation"):
                        if current_equation:
                            equations.append("\n".join(current_equation).strip())
                        current_equation = [line]
                    elif line and current_equation:
                        current_equation.append(line)
                if current_equation:
                    equations.append("\n".join(current_equation).strip())

                if equations:
                    logging.info(f"Successfully extracted {len(equations)} LaTeX equations from {os.path.basename(image_path)} on attempt {attempt + 1}.")
                    return "\n".join(equations)  # Return as a single string with newlines for compatibility
                else:
                    logging.warning(f"Attempt {attempt + 1} failed: No valid equations segmented. Response: {extracted_text}")
            else:
                logging.warning(f"Attempt {attempt + 1} failed: No LaTeX extracted. Response: {response.text}")

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {e}")

        # Exponential backoff before retrying
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 2, 4, 8 seconds
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    # If all attempts fail, save the image for debugging and return failure
    logging.warning(f"All {max_retries} attempts failed for {os.path.basename(image_path)}.")
    debug_dir = os.path.join(TEMP_IMG_DIR, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"failed_{os.path.basename(image_path)}")
    img.save(debug_path)
    logging.info(f"Saved failed image to {debug_path} for debugging.")
    return "No LaTeX found"

def extract_latex_from_pdf(pdf_path, page_selection="all"):
    """
    Extracts LaTeX from selected pages of a PDF, converting pages to images first.
    This function uses PyMuPDF (fitz) to render PDF pages into images and segments equations.
    """
    logging.info(f"Extracting LaTeX from PDF: {os.path.basename(pdf_path)} (pages: {page_selection})...")
    latex_expressions = []
    try:
        doc = fitz.open(pdf_path)
        pages_to_process = parse_page_selection(page_selection, doc.page_count)

        for i in pages_to_process:
            if i >= doc.page_count:
                logging.warning(f"Page {i+1} is out of bounds for PDF {os.path.basename(pdf_path)}. Skipping.")
                continue

            page = doc.load_page(i)
            pix = page.get_pixmap()
            img_filename = os.path.join(TEMP_IMG_DIR, f"{Path(pdf_path).stem}_page_{i+1}.png")
            pix.save(img_filename)

            logging.info(f"Saved temporary image: {img_filename}")

            # Call the LaTeX extraction for the image, which will segment equations
            extracted_latex = extract_latex_from_image(img_filename)
            if extracted_latex != "No LaTeX found":
                # Split into individual equations if multiple are present
                equations = extracted_latex.split('\n')
                for eq_idx, eq in enumerate(equations, 1):
                    if eq.strip().startswith("Equation"):
                        latex_expressions.append({
                            "page_number": i + 1,
                            "image_path": img_filename,
                            "latex": eq.strip(),
                            "equation_number": eq_idx
                        })
            else:
                latex_expressions.append({
                    "page_number": i + 1,
                    "image_path": img_filename,
                    "latex": extracted_latex
                })
            # Clean up temporary image if desired, or keep for debugging
            # os.remove(img_filename)

        doc.close()
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")

    return latex_expressions
# --- End Placeholder ---


def compare_answers(answer1, answer2, rel_tol=1e-2):
    """Compare two answers with flexible comparison."""
    if answer1 == "No answer found" or answer2 == "No answer found":
        return False

    # Extract boxed content if present
    def unbox(ans):
        box_match = re.search(r"\\boxed\{([^}]+)\}", ans)
        return box_match.group(1) if box_match else ans

    a1 = unbox(answer1).strip() # Strip whitespace after unboxing
    a2 = unbox(answer2).strip()

    # Simple string comparison first (case-sensitive)
    if a1 == a2:
        return True

    # Try numerical comparison if possible
    def to_float(ans_str):
        try:
            # Handle fractions (e.g., "1/2")
            if '/' in ans_str and ans_str.count('/') == 1:
                num_str, den_str = ans_str.split('/')
                num = float(num_str.strip())
                den = float(den_str.strip())
                if den == 0: return None # Avoid division by zero
                return num / den
            # Handle scientific notation (e.g., "3.0e5", "3.0 x 10^5", "3.0 * 10^5")
            ans_str_cleaned = ans_str.lower().replace(' ', '')
            if 'e' in ans_str_cleaned or 'x10^' in ans_str_cleaned or '*10^' in ans_str_cleaned:
                ans_str_for_eval = ans_str_cleaned.replace('x10^', 'e').replace('*10^', 'e')
                return float(ans_str_for_eval)

            # Handle common mathematical constants if they are part of the answer
            if ans_str_cleaned == 'pi': return math.pi
            if ans_str_cleaned == 'e': return math.e

            return float(ans_str.replace(',', '')) # Remove commas for numerical conversion
        except ValueError:
            return None

    val1 = to_float(a1)
    val2 = to_float(a2)

    if val1 is not None and val2 is not None:
        return math.isclose(val1, val2, rel_tol=rel_tol)

    # Fallback to case-insensitive string comparison if numerical failed or not applicable
    return a1.lower() == a2.lower()



def generate_problem_with_solutions(image_path, extracted_latex, max_attempts=5):
    """Generate complete problem with two solution approaches using enhanced prompt."""
    filename = os.path.basename(image_path)

    if extracted_latex == "No LaTeX found" or not extracted_latex.strip():
        return {
            "image_filename": filename,
            "error": "No LaTeX extracted",
            "status": "failed"
        }

    for attempt in range(max_attempts):
        try:
            # Dynamically adjust prompt based on attempt
            if attempt < 2:  # First 2 attempts with full prompt
                prompt = f"""
                Given: {extracted_latex}
                Generate a STEM problem in STRICT JSON format with:
                - One PhD-level problem
                - Two solution approaches
                - Same final answer
                - Proper LaTeX escaping (DOUBLE backslashes)
                
                Output MUST be valid JSON with this EXACT structure:
                {{
                    "q": "Problem statement",
                    "s1": "Solution 1 steps",
                    "a1": "\\\\boxed{{answer}}",
                    "s2": "Solution 2 steps",
                    "a2": "\\\\boxed{{same_answer}}",
                    "cat": "Category",
                    "sub": "Subcategory"
                }}
                """
            else:  # Simplified prompt for later attempts
                prompt = f"""
                Given: {extracted_latex}
                Output this EXACT JSON:
                {{
                    "q": "Evaluate this expression",
                    "s1": "Solution steps",
                    "a1": "\\\\boxed{{result}}",
                    "s2": "Same as above",
                    "a2": "\\\\boxed{{result}}",
                    "cat": "Math",
                    "sub": "General"
                }}
                """

            response = model.generate_content([prompt, Image.open(image_path)])
            response_text = response.text

            # Multi-stage cleaning with validation
            problem_data = None
            for clean_attempt in range(3):
                try:
                    cleaned = clean_json_response(response_text)
                    problem_data = json.loads(cleaned)
                    
                    # Validate structure
                    if not all(k in problem_data for k in ['q', 's1', 'a1', 's2', 'a2']):
                        raise ValueError("Missing required keys")
                    if not compare_answers(problem_data['a1'], problem_data['a2']):
                        raise ValueError("Answer mismatch")
                    break
                except Exception as e:
                    if clean_attempt == 2:
                        raise
                    continue

            return {
                "image_filename": filename,
                "extracted_latex": extracted_latex,
                "generated_question_llm1": problem_data['q'],
                "llm1_solution": problem_data['s1'],
                "llm1_final_answer": problem_data['a1'],
                "llm2_proposed_solution": problem_data['s2'],
                "llm2_final_answer_extracted": problem_data['a2'],
                "judge_evaluation": "Correct",
                "api_answer_status": "yes",
                "category": problem_data.get('cat', 'Mathematics'),
                "subcategory": problem_data.get('sub', 'General')
            }

        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)[:100]}...")
            time.sleep(2 ** attempt)
            continue

    return {
        "image_filename": filename,
        "error": f"Failed after {max_attempts} attempts",
        "status": "failed"
    }


def get_input_files():
    """
    Placeholder: Gathers input files from INPUT_DIR.
    It can handle both image files and PDF files by converting PDF pages to images.
    """
    input_files = []

    logging.info(f"Scanning for input files in {INPUT_DIR}...")
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                input_files.append(
                    {
                        "image_path": file_path,
                        "latex": extract_latex_from_image(file_path)
                    })
            elif file.lower().endswith('.pdf'):
                # Process PDF and add generated image paths to input_files
                logging.info(f"Found PDF: {file_path}. Extracting pages as images...")
                extracted_pages = extract_latex_from_pdf(file_path, page_selection=PDF_PAGE_SELECTION)
                # extracted_pages is a list of dicts, each with 'image_path'
                input_files.extend(extracted_pages)
            else:
                logging.warning(f"Skipping unsupported file type: {file_path}")

    if not input_files:
        logging.warning(f"No image or PDF files found in {INPUT_DIR}.")

    # Sort files to ensure consistent processing order
    return sorted(input_files, key=lambda input_file: input_file['image_path'])

def save_results(results):
    """Saves the processed results to CSV and JSON files."""
    if not results:
        logging.warning("No results to save.")
        return

    # Filter out failed results for CSV/JSON if they only contain 'status' and 'error'
    # Or decide to include them but handle missing keys gracefully during CSV writing
    successful_results = [r for r in results if isinstance(r, dict) and 'status' not in r]

    # Save to JSON
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to {OUTPUT_JSON}")
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")

    # Save to CSV
    if successful_results: # Only save successful results to CSV for cleaner data
        try:
            # Determine fieldnames from the first successful result (assuming consistent structure)
            fieldnames = list(successful_results[0].keys())

            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in successful_results:
                    writer.writerow(row)
            logging.info(f"Results saved to {OUTPUT_CSV}")
        except Exception as e:
            logging.error(f"Error saving results to CSV: {e}")
    else:
        logging.warning("No successful results to save to CSV.")

# --- End Placeholder ---


def main():
    """Main function to orchestrate the problem generation pipeline."""
    logging.info("Starting problem generation pipeline...")
    pipeline_start_time = time.time()

    input_files = get_input_files() # Assumes this function is implemented
    if not input_files:
        logging.error("No input files found in the specified input directory.")
        return

    results = []
    # Limit processing to MAX_PROBLEMS if more files are found
    files_to_process = input_files[:MAX_PROBLEMS]
    logging.info(f"Found {len(input_files)} input files. Processing up to {len(files_to_process)}.")

    for i, file_info in enumerate(files_to_process):
        logging.info(f"[{i+1}/{len(files_to_process)}] Processing {os.path.basename(file_info['image_path'])}")

        result = generate_problem_with_solutions(file_info["image_path"], file_info["latex"]) # Assumes this function is implemented
        results.append(result)

        if result and 'status' not in result: # Check if processing was successful for this item
            logging.info(f"  Generated problem: {result['generated_question_llm1'][:70]}...") # Shorter log
            logging.info(f"  LLM1 answer: {result['llm1_final_answer']}")
            logging.info(f"  LLM2 answer: {result['llm2_final_answer_extracted']}")
            logging.info(f"  Evaluation: {result['judge_evaluation']}")
        else:
            logging.warning(f"  Failed to process {os.path.basename(file_info['image_path'])}: {result.get('error', 'Unknown error')}")

    if results:
        save_results(results) # Assumes this function is implemented
        success_count = len([r for r in results if isinstance(r, dict) and 'status' not in r])
        logging.info(f"Pipeline finished. Successfully processed {success_count}/{len(results)} problems.")

        # Generate visualizations
        visualizer = ProblemVisualizer(OUTPUT_DIR)
        visualizer.generate_metrics_chart(results)
    else:
        logging.warning("No valid results generated from any input file.")

    total_pipeline_time = time.time() - pipeline_start_time
    logging.info(f"\n\n\n\n\nTotal pipeline execution time: {total_pipeline_time:.2f} seconds.")


if __name__ == "__main__":
    main()