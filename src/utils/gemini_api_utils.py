# src/utils/gemini_api_utils.py
import google.generativeai as genai
from google.api_core import exceptions
from PIL import Image
from pathlib import Path
import io
import time
import json
import re
from typing import Dict, Any, List, Tuple

# Import the initialized model instance from gemini_api
from src.utils.gemini_api import model # Ensure 'model' is imported

# --- API Configuration and Utilities ---
def configure_gemini():
    """
    No-op function for consistency, as genai.configure is now handled in src/utils/gemini_api.py.
    This function can remain if older calls expect it, but its actual work is elsewhere.
    """
    pass # Configuration is handled in src/utils/gemini_api.py on import

def _extract_and_clean_json(text: str) -> str:
    """
    Extracts a JSON string from a larger text, handling markdown code blocks
    and attempts to fix common JSON issues like trailing commas or unescaped newlines.
    """
    # 1. Strip markdown code block fences
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```', '', text)

    # 2. Attempt to escape common unescaped characters within JSON values (simplified)
    # This is a heuristic and might need adjustment based on common model errors
    # Specifically target unescaped backslashes, which are common in LaTeX output
    # WARNING: This can be tricky and might over-escape. Only apply if necessary.
    # A safer approach is to ask the model to ensure valid JSON or parse leniently.
    # For now, let's focus on basic cleanup.
    # text = text.replace('\\', '\\\\') # This is often too aggressive, better to handle during parsing if needed

    # 3. Try to extract content between the first { and last }
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    except Exception:
        pass # If extraction fails, use the raw text for subsequent steps

    return text

def _robust_json_load(json_string: str, call_description: str = "JSON parse") -> Dict[str, Any]:
    """
    Attempts to load a JSON string, handling common parsing errors.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"    [X] JSONDecodeError during {call_description}: {e}")
        print(f"        Problematic JSON string (first 500 chars): {json_string[:500]}...")
        # Attempt a more lenient parsing if simple json.loads fails
        try:
            # Basic cleanup before re-attempting parse for common issues
            cleaned_string = re.sub(r',\s*}', '}', json_string) # Remove trailing commas before '}'
            cleaned_string = re.sub(r',\s*]', ']', cleaned_string) # Remove trailing commas before ']'
            # Try to fix unescaped newlines within strings (basic heuristic)
            cleaned_string = re.sub(r'(?<!\\)\n', r'\\n', cleaned_string) # Replace unescaped newlines with '\n'

            return json.loads(cleaned_string)
        except json.JSONDecodeError as e_cleaned:
            print(f"    [X] JSONDecodeError even after cleanup for {call_description}: {e_cleaned}")
            print(f"        Cleaned JSON string (first 500 chars): {cleaned_string[:500]}...")
            return {} # Return empty dict on severe parsing failure
        except Exception as e_other:
            print(f"    [X] Other error during JSON cleanup/re-parse for {call_description}: {e_other}")
            return {}
    except Exception as e:
        print(f"    [X] Unexpected error during {call_description}: {e}")
        return {}

def load_image_from_path(image_path: Path) -> Image.Image:
    """Loads an image from a given path."""
    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"    [X] Error: Image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"    [X] Error loading image {image_path}: {e}")
        raise

def call_gemini_api_with_retries(
    model_instance: genai.GenerativeModel,
    prompt_content: Any, # Can be str, list of Parts, etc.
    call_description: str = "API call",
    max_retries: int = 5,
    base_delay: float = 2.0
) -> Tuple[Any, int, int, float]: # response, prompt_tokens, candidate_tokens, duration
    """
    Calls the Gemini API with exponential back-off and retries for common transient errors.
    Returns the response object, prompt_tokens, candidate_tokens, and duration.
    """
    attempt = 0
    prompt_tokens = 0
    candidate_tokens = 0
    duration = 0.0
    while attempt < max_retries:
        attempt += 1
        try:
            start_time = time.time()
            response = model_instance.generate_content(prompt_content)
            end_time = time.time()
            duration = end_time - start_time

            # Extract token counts safely
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                candidate_tokens = response.usage_metadata.candidates_token_count
            else:
                print(f"    [!] No usage_metadata found for {call_description} attempt {attempt}.")

            # Check for empty or blocked responses
            if not response.candidates:
                print(f"    [X] {call_description} attempt {attempt}: Response has no candidates. Retrying...")
                raise ValueError("No candidates in response")
            if response.candidates[0].finish_reason == genai.protos.FinishReason.SAFETY:
                print(f"    [X] {call_description} attempt {attempt}: Response blocked by safety settings. Skipping retries for safety block.")
                return None, 0, 0, 0.0 # Do not retry on safety blocks
            if not response.candidates[0].content.parts:
                print(f"    [X] {call_description} attempt {attempt}: Response candidate has no content parts. Retrying...")
                raise ValueError("No content parts in candidate")

            print(f"    [+] {call_description} successful (attempt {attempt}, took {duration:.2f}s). Tokens: (P:{prompt_tokens}, C:{candidate_tokens})")
            return response, prompt_tokens, candidate_tokens, duration

        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable,
                exceptions.InternalServerError, exceptions.TooManyRequests,
                ValueError) as e: # ValueError for custom checks like "No candidates"
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) # Exponential back-off
                print(f"    [!] {call_description} attempt {attempt} failed ({e}). Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"    [X] {call_description} failed after {max_retries} attempts ({e}). Giving up.")
                return None, 0, 0, 0.0
        except Exception as e:
            print(f"    [X] An unexpected error occurred during {call_description}: {e}")
            return None, 0, 0, 0.0
    return None, 0, 0, 0.0 # Should not be reached, but for type hinting

# --- Gemini Specific Q&A Generation Functions ---

def gemini_extract_latex_from_image(image_path: Path, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Extracts LaTeX expressions from an image using the Gemini API, with retries.
    Returns a dictionary including the extracted LaTeX string and API metrics.
    """
    try:
        img = load_image_from_path(image_path)
        prompt_parts = [
            "Extract all mathematical equations and expressions from this image. "
            "Provide them as a single, concatenated string of LaTeX code. "
            "If multiple expressions are present, separate them with a double backslash `\\\\`. "
            "Do not include any other text or explanation, just the LaTeX code.",
            img
        ]
        print(f"    [>] Sending image {image_path.name} to Gemini for LaTeX extraction...")

        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description=f"LaTeX extraction for {image_path.name}"
        )

        extracted_latex = "Error: No LaTeX extracted."
        if response and response.candidates and response.candidates[0].content.parts:
            extracted_latex = response.candidates[0].content.parts[0].text.strip()
            # Basic cleanup: remove leading/trailing markdown, if any
            extracted_latex = re.sub(r'^\s*`{3}latex\s*|\s*`{3}\s*$', '', extracted_latex, flags=re.DOTALL)
            extracted_latex = extracted_latex.strip()

        return {
            "extracted_latex": extracted_latex,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error extracting LaTeX for {image_path.name}: {e}")
        return {
            "extracted_latex": f"Error: {e}",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_generate_qna_and_solution(image_path: Path, extracted_latex : str, model_instance : genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates a math Q&A problem, its detailed step-by-step solution, and a canonical final answer
    from an image and extracted LaTeX using the Gemini API, with retries.
    Returns a dictionary including Q&A data and API metrics.
    """
    try:
        img = load_image_from_path(image_path)
        prompt_parts = [
            "Given this image and the extracted LaTeX expressions, generate a "
            "challenging math problem, a detailed step-by-step solution, "
            "a single canonical final answer, an array of critical expressions in LaTeX, "
            "an array of key topics, and an array of critical steps. "
            "Format the output as a JSON object with the following keys: "
            "'question', 'solution', 'final_answer' (string in LaTeX if applicable), "
            "'critical_expressions' (array of LaTeX strings), 'topics' (array of strings), "
            "and 'critical_steps' (array of strings).",
            f"Extracted LaTeX:\n{extracted_latex}",
            img
        ]
        print(f"    [>] Sending image {image_path.name} to Gemini for Q&A generation...")

        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description=f"Q&A generation for {image_path.name}"
        )

        json_string = ""
        if response and response.candidates and response.candidates[0].content.parts:
            raw_response_text = response.candidates[0].content.parts[0].text
            json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string

        # Use the robust JSON loader here
        qna_data = _robust_json_load(json_string, call_description=f"Q&A JSON parse for {image_path.name}")

        generated_question = "Error: Failed to parse Q&A from image."
        solution = f"Error: {json_string}" # Keep raw JSON for debugging if parsing fails
        final_answer = "N/A"
        critical_expressions = []
        topics = []
        critical_steps = []

        if qna_data: # Check if parsing was successful
            # Basic validation based on the requested output format in the prompt
            required_keys = ["question", "solution", "final_answer", "critical_expressions", "topics", "critical_steps"]
            if all(key in qna_data for key in required_keys):
                generated_question = qna_data.get("question", generated_question)
                solution = qna_data.get("solution", solution)
                final_answer = qna_data.get("final_answer", final_answer)
                critical_expressions = qna_data.get("critical_expressions", critical_expressions)
                topics = qna_data.get("topics", topics)
                critical_steps = qna_data.get("critical_steps", critical_steps)
            else:
                print(f"    [X] Missing required keys in Gemini's Q&A JSON response for {image_path.name}.")

        return {
            "generated_question": generated_question,
            "solution": solution,
            "final_answer": final_answer,
            "critical_expressions": critical_expressions,
            "topics": topics,
            "critical_steps": critical_steps,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error generating Q&A for {image_path.name}: {e}")
        return {
            "generated_question": f"Error during Q&A generation: {e}",
            "solution": "N/A",
            "final_answer": "N/A",
            "critical_expressions": [],
            "topics": [],
            "critical_steps": [],
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_generate_solution_for_question(question: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates a step-by-step solution for a given mathematical question using the Gemini API.
    This is intended to produce the LLM2 proposed solution.
    Returns a dictionary including the solution and API metrics.
    """
    try:
        prompt_parts = [
            f"Solve the following mathematical question step-by-step. Provide a detailed solution.\nQuestion: {question}",
        ]
        print(f"    [>] Sending question to Gemini for LLM2 solution generation...")
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description="LLM2 Solution generation"
        )
        solution_text = "N/A: No solution generated by LLM2."
        if response and response.candidates and response.candidates[0].content.parts:
            solution_text = response.candidates[0].content.parts[0].text.strip()

        return {
            "solution_text": solution_text,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error generating LLM2 solution for question: {e}")
        return {
            "solution_text": f"Error generating LLM2 solution: {e}",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_extract_final_answer_from_solution(solution_text: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Extracts the concise final numerical or mathematical answer from a detailed solution text.
    Returns a dictionary including the extracted answer and API metrics.
    """
    if not solution_text or solution_text.lower() in ["n/a", "error", "n/a: no solution generated by llm2."]:
        return {
            "extracted_answer": "N/A",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

    try:
        prompt_parts = [
            f"From the following mathematical solution, extract ONLY the final concise answer. "
            f"If the answer is numerical, provide only the number. If it's a mathematical expression, provide it in LaTeX. "
            f"Do not include any other text or explanation. If no clear final answer is present, output 'N/A'.\nSolution: {solution_text}",
        ]
        print(f"    [>] Extracting final answer from solution...")
        response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
            model_instance=model_instance,
            prompt_content=prompt_parts,
            call_description="Final answer extraction"
        )
        extracted_answer = "N/A"
        if response and response.candidates and response.candidates[0].content.parts:
            extracted_answer = response.candidates[0].content.parts[0].text.strip()
            # Basic cleanup: sometimes it might wrap in markdown even if not requested
            if extracted_answer.startswith("```") and extracted_answer.endswith("```"):
                extracted_answer = extracted_answer.strip("` \n")
            if extracted_answer.lower() == "n/a":
                extracted_answer = "N/A" # Ensure consistent "N/A" capitalization

        return {
            "extracted_answer": extracted_answer,
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidate_tokens,
            "api_call_duration": duration
        }
    except Exception as e:
        print(f"    [X] Error extracting final answer: {e}")
        return {
            "extracted_answer": "N/A",
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

def gemini_evaluate_solution_as_judge(question: str, llm1_solution: str, llm2_solution: str) -> Dict[str, Any]:
    """
    Evaluates LLM2's solution against LLM1's using Gemini as a judge.
    Returns a dictionary including the evaluation result and API metrics.
    """
    prompt_parts = [
        f"You are an expert mathematical judge. Here is a question:\nQuestion: {question}\n\n"
        f"Here is a correct ground truth solution (LLM1):\nLLM1 Solution: {llm1_solution}\n\n"
        f"Here is a proposed solution (LLM2):\nLLM2 Solution: {llm2_solution}\n\n"
        "Compare LLM2's solution to LLM1's. Is LLM2's solution correct and equivalent to LLM1's? "
        "Respond only with 'Correct' or 'Incorrect'. No explanations.",
    ]
    print(f"    [>] Evaluating solution with Gemini judge...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Solution evaluation"
    )
    evaluation_result = "Error: No response from judge."
    if response and response.candidates and response.candidates[0].content.parts:
        evaluation_result = response.candidates[0].content.parts[0].text.strip()

    return {
        "evaluation_result": evaluation_result,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_determine_category_and_subcategory(question: str, solution: str, extracted_math: str) -> Dict[str, Any]:
    """
    Determines the category and subcategory of the math problem using Gemini.
    Returns a dictionary including category data and API metrics.
    """
    prompt_parts = [
        f"Based on the following mathematical question, solution, and extracted math, categorize the problem.\n"
        f"Question: {question}\nSolution: {solution}\nExtracted Math: {extracted_math}\n\n"
        "Provide a single, concise category and subcategory (e.g., 'Algebra', 'Linear Equations'; 'Calculus', 'Derivatives'). "
        "Format your response as a JSON object with keys `category` and `subcategory`."
    ]
    print(f"    [>] Determining category with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Category determination"
    )
    json_string = ""
    if response and response.candidates and response.candidates[0].content.parts:
        raw_response_text = response.candidates[0].content.parts[0].text
        json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string

    # Use the robust JSON loader here
    category_data = _robust_json_load(json_string, call_description="Category JSON parse")

    category = "Uncategorized"
    subcategory = "General"
    if category_data:
        category = category_data.get("category", "Uncategorized")
        subcategory = category_data.get("subcategory", "General")
    else:
        print(f"    [X] Gemini returned malformed JSON for category, or parsing failed.")

    return {
        "category": category,
        "subcategory": subcategory,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_topics(question: str, solution: str, extracted_math: str) -> Dict[str, Any]:
    """
    Generates a list of topics related to the math problem using Gemini.
    Returns a dictionary including the list of topics and API metrics.
    """
    prompt_parts = [
        f"Identify key mathematical topics covered in this problem:\n"
        f"Question: {question}\nSolution: {solution}\nExtracted Math: {extracted_math}\n\n"
        "Provide a comma-separated list of topics (e.g., 'Differentiation', 'Integration', 'Trigonometry')."
    ]
    print(f"    [>] Generating topics with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Topic generation"
    )
    response_text = ""
    if response and response.candidates and response.candidates[0].content.parts:
        response_text = response.candidates[0].content.parts[0].text
    topics_list = [topic.strip() for topic in response_text.split(',') if topic.strip()]

    return {
        "topics_list": topics_list,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_determine_api_answer_status(llm2_proposed_solution: str) -> Dict[str, Any]:
    """
    Determines if LLM2 provided a valid answer.
    Returns a dictionary including the status and API metrics.
    """
    prompt_parts = [
        f"Review the following solution: '{llm2_proposed_solution}'. Does it appear to be a legitimate attempt at solving a math problem, or is it an empty/error/irrelevant response? Respond with 'Yes' if it's a valid attempt, or 'No' otherwise.",
    ]
    print(f"    [>] Determining API answer status with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="API answer status determination"
    )
    status = "0" # Default to "0" as previously
    if response and response.candidates and response.candidates[0].content.parts:
        status = response.candidates[0].content.parts[0].text.strip().lower()
        if status == "yes":
            status = "1"
        elif status == "no":
            status = "0"

    return {
        "status": status,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_alternate_answers(canonical_answer: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates alternate correct answers if possible, using Gemini.
    Returns a dictionary including the list of alternate answers and API metrics.
    """
    if not canonical_answer or canonical_answer.lower() in ["n/a", "error"]:
        return {
            "alternate_answers": [],
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "api_call_duration": 0.0
        }

    prompt_parts = [
        f"Given the canonical final answer: '{canonical_answer}'. "
        "Provide 1-2 mathematically equivalent alternate representations or forms of this answer. "
        "If no meaningful alternate forms exist (e.g., for simple numbers), state 'None'. "
        "Format as a comma-separated list of LaTeX strings.",
    ]
    print(f"    [>] Generating alternate answers with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model_instance, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Alternate answers generation"
    )
    response_text = ""
    alternate_answers_list = []
    if response and response.candidates and response.candidates[0].content.parts:
        response_text = response.candidates[0].content.parts[0].text
        if "none" not in response_text.lower() and response_text.strip():
            alternate_answers_list = [ans.strip() for ans in response_text.split(',') if ans.strip()]

    return {
        "alternate_answers": alternate_answers_list,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }

def gemini_generate_critical_expressions_and_steps(question: str, llm1_solution: str, model_instance: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Generates critical mathematical expressions and solution steps using Gemini.
    Returns a dictionary including critical expressions/steps and API metrics.
    """
    prompt_parts = [
        f"From the following question and its solution, identify:\n"
        f"1. Key mathematical expressions that define the problem or its core parts.\n"
        f"2. The most critical steps or transformations in the solution process.\n"
        f"Question: {question}\nSolution: {llm1_solution}\n\n"
        "Format your response as a JSON object with two keys: `critical_expressions` (a list of LaTeX strings) and `critical_steps` (a list of concise English descriptions of steps)."
    ]
    print(f"    [>] Generating critical expressions and steps with Gemini...")
    response, prompt_tokens, candidate_tokens, duration = call_gemini_api_with_retries(
        model_instance=model_instance, # Using the globally imported 'model' instance
        prompt_content=prompt_parts,
        call_description="Critical expressions and steps generation"
    )
    json_string = ""
    critical_expressions = []
    critical_steps = []

    if response and response.candidates and response.candidates[0].content.parts:
        raw_response_text = response.candidates[0].content.parts[0].text
        json_string = _extract_and_clean_json(raw_response_text) # Clean the JSON string

    # Use the robust JSON loader here
    critical_data = _robust_json_load(json_string, call_description="Critical data JSON parse")

    if critical_data:
        critical_expressions = critical_data.get("critical_expressions", [])
        critical_steps = critical_data.get("critical_steps", [])
    else:
        print(f"    [X] Gemini returned malformed JSON for critical data, or parsing failed.")

    return {
        "critical_expressions": critical_expressions,
        "critical_steps": critical_steps,
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "api_call_duration": duration
    }