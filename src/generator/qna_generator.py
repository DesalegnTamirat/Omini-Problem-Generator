# src/generator/qna_generator.py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import google.generativeai as genai

# Import utility functions from gemini_api_utils
from src.utils import gemini_api_utils

class QnAGenerator:
    def __init__(self, model: genai.GenerativeModel, output_dir: Path):
        """
        Initializes the QnA Generator.
        Args:
            model: The initialized Gemini GenerativeModel instance to use for API calls.
            output_dir: The base directory where final Q&A data will be saved.
        """
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        self.qna_data: List[Dict[str, Any]] = [] # To accumulate all generated Q&A records

    def process_image_for_qna(self, image_path: Path) -> Dict[str, Any]:
        """
        Processes a single image to generate a comprehensive Q&A record,
        including solution, various metadata, and evaluation.
        Aggregates token usage and time spent across all API calls for this question.

        Args:
            image_path: The path to the image file to process.

        Returns:
            A dictionary containing all generated Q&A data and API metrics for the image.
        """
        print(f"  [>>] Processing image {image_path.name} for Q&A generation...")

        # Initialize overall metrics for this single question's processing
        total_prompt_tokens_per_question = 0
        total_candidate_tokens_per_question = 0
        total_api_time_per_question = 0.0

        try:
            # 1. Extract LaTeX expressions from the image (LLM0)
            latex_extraction_result = gemini_api_utils.gemini_extract_latex_from_image(image_path, self.model)
            extracted_latex = latex_extraction_result.get("extracted_latex", "N/A")
            total_prompt_tokens_per_question += latex_extraction_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += latex_extraction_result.get("candidates_token_count", 0)
            total_api_time_per_question += latex_extraction_result.get("api_call_duration", 0.0)
            print(f"    [->] Extracted LaTeX: {'<No LaTeX>' if extracted_latex == 'N/A' or not extracted_latex.strip() else extracted_latex[:70] + '...' if len(extracted_latex) > 70 else extracted_latex}")


            # 2. Generate Q&A, solution, final answer, critical expressions, topics, steps (LLM1 - Ground Truth)
            qna_gen_result = gemini_api_utils.gemini_generate_qna_and_solution(image_path, extracted_latex, self.model)
            generated_question_llm1 = qna_gen_result.get("generated_question", "Error")
            llm1_solution = qna_gen_result.get("solution", "Error")
            llm1_final_answer = qna_gen_result.get("final_answer", "N/A")
            llm1_critical_expressions = qna_gen_result.get("critical_expressions", [])
            topics = qna_gen_result.get("topics", [])
            llm1_critical_steps = qna_gen_result.get("critical_steps", [])
            total_prompt_tokens_per_question += qna_gen_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += qna_gen_result.get("candidates_token_count", 0)
            total_api_time_per_question += qna_gen_result.get("api_call_duration", 0.0)
            print(f"    [->] LLM1 Q&A Generated. Question: {generated_question_llm1[:70]}...")


            # 3. Generate LLM2 proposed solution (independent solution for comparison)
            llm2_solution_result = gemini_api_utils.gemini_generate_solution_for_question(generated_question_llm1, self.model)
            llm2_proposed_solution = llm2_solution_result.get("solution_text", "Error")
            total_prompt_tokens_per_question += llm2_solution_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += llm2_solution_result.get("candidates_token_count", 0)
            total_api_time_per_question += llm2_solution_result.get("api_call_duration", 0.0)
            print(f"    [->] LLM2 Solution Generated.")


            # 4. Extract final answer from LLM2 solution
            llm2_final_answer_result = gemini_api_utils.gemini_extract_final_answer_from_solution(llm2_proposed_solution, self.model)
            llm2_final_answer_extracted = llm2_final_answer_result.get("extracted_answer", "N/A")
            total_prompt_tokens_per_question += llm2_final_answer_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += llm2_final_answer_result.get("candidates_token_count", 0)
            total_api_time_per_question += llm2_final_answer_result.get("api_call_duration", 0.0)
            print(f"    [->] LLM2 Final Answer Extracted: {llm2_final_answer_extracted}")


            # 5. Evaluate LLM2's solution against LLM1's (Gemini as judge)
            evaluation_result = gemini_api_utils.gemini_evaluate_solution_as_judge(generated_question_llm1, llm1_solution, llm2_proposed_solution)
            judge_evaluation = evaluation_result.get("evaluation_result", "Error")
            total_prompt_tokens_per_question += evaluation_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += evaluation_result.get("candidates_token_count", 0)
            total_api_time_per_question += evaluation_result.get("api_call_duration", 0.0)
            print(f"    [->] Judge Evaluation: {judge_evaluation}")


            # 6. Determine API answer status (valid attempt or not)
            api_answer_status_result = gemini_api_utils.gemini_determine_api_answer_status(llm2_proposed_solution)
            api_answer_status = api_answer_status_result.get("status", "0") # Default to "0"
            total_prompt_tokens_per_question += api_answer_status_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += api_answer_status_result.get("candidates_token_count", 0)
            total_api_time_per_question += api_answer_status_result.get("api_call_duration", 0.0)
            print(f"    [->] LLM2 Answer Status: {'Valid' if api_answer_status == '1' else 'Invalid'}")


            # 7. Generate alternate answers for LLM1's final answer
            alternate_answers_result = gemini_api_utils.gemini_generate_alternate_answers(llm1_final_answer, self.model)
            alternate_answers = alternate_answers_result.get("alternate_answers", [])
            total_prompt_tokens_per_question += alternate_answers_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += alternate_answers_result.get("candidates_token_count", 0)
            total_api_time_per_question += alternate_answers_result.get("api_call_duration", 0.0)
            print(f"    [->] Alternate Answers: {', '.join(alternate_answers) if alternate_answers else 'None'}")


            # 8. Determine Category and Subcategory
            category_result = gemini_api_utils.gemini_determine_category_and_subcategory(generated_question_llm1, llm1_solution, extracted_latex)
            category = category_result.get("category", "Uncategorized")
            subcategory = category_result.get("subcategory", "General")
            total_prompt_tokens_per_question += category_result.get("prompt_token_count", 0)
            total_candidate_tokens_per_question += category_result.get("candidates_token_count", 0)
            total_api_time_per_question += category_result.get("api_call_duration", 0.0)
            print(f"    [->] Category: {category}, Subcategory: {subcategory}")


            return {
                "image_filename": image_path.name,
                "extracted_latex": extracted_latex,
                "generated_question_llm1": generated_question_llm1,
                "llm1_solution": llm1_solution,
                "llm1_final_answer": llm1_final_answer,
                "llm2_proposed_solution": llm2_proposed_solution,
                "llm2_final_answer_extracted": llm2_final_answer_extracted,
                "judge_evaluation": judge_evaluation,
                "api_answer_status": api_answer_status, # "1" for valid, "0" for invalid
                "llm1_critical_expressions": llm1_critical_expressions,
                "topics": topics,
                "llm1_critical_steps": llm1_critical_steps,
                "alternate_answers": alternate_answers,
                "category": category,
                "subcategory": subcategory,
                "total_prompt_tokens_per_question": total_prompt_tokens_per_question,
                "total_candidate_tokens_per_question": total_candidate_tokens_per_question,
                "total_api_time_per_question": total_api_time_per_question
            }

        except Exception as e:
            print(f"  [X] Failed to process {image_path.name} due to an error: {e}")
            # Return an error dictionary with aggregated totals even on error
            return {
                "image_filename": image_path.name,
                "extracted_latex": "Error",
                "generated_question_llm1": f"Error processing image: {e}",
                "llm1_solution": "Error",
                "llm1_final_answer": "Error",
                "llm2_proposed_solution": "Error",
                "llm2_final_answer_extracted": "Error",
                "judge_evaluation": "Error",
                "api_answer_status": "0", # Mark as invalid attempt
                "llm1_critical_expressions": [],
                "topics": [],
                "llm1_critical_steps": [],
                "alternate_answers": [],
                "category": "Error",
                "subcategory": "Error",
                "total_prompt_tokens_per_question": total_prompt_tokens_per_question,
                "total_candidate_tokens_per_question": total_candidate_tokens_per_question,
                "total_api_time_per_question": total_api_time_per_question
            }

    def save_qna_data(self, data: List[Dict[str, Any]], filename: str = "final_qna_dataset.csv"):
        """
        Saves the accumulated Q&A data to a CSV file.
        Args:
            data: The list of Q&A dictionaries to save.
            filename: The name of the CSV file to save.
        """
        if not data:
            print("[!] No Q&A data to save.")
            return

        df = pd.DataFrame(data)
        output_filepath = self.output_dir / filename

        # Convert list columns to JSON strings for CSV compatibility
        for col in ['critical_expressions', 'topics', 'critical_steps', 'alternate_answers']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x) # Ensure it's a list/dict before dumping

        try:
            df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"[✓] Q&A data saved to {output_filepath}")
        except Exception as e:
            print(f"[X] Error saving Q&A data to CSV: {e}")
            print(f"[!] Data might be too complex for direct CSV, consider JSON output only.")