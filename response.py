import google.generativeai as genai
from src.utils.gemini_api import model
from src.config.settings import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

response = model.generate_content("Explain quantum physics in simple terms")

# # Print basic text
# print("Response Text:\n", response.text)

# # Inspect parts
# print("\nParts:")
# for i, part in enumerate(response.parts):
#     print(f"Part {i}: {part.text if hasattr(part, 'text') else part}")

# # Candidates
# if hasattr(response, 'candidates'):
#     print("\nCandidates:")
#     for i, candidate in enumerate(response.candidates):
#         print(f"Candidate {i}:")
#         print("  Finish reason:", candidate.finish_reason)
#         print("  Content:", candidate.content.parts[0].text)

# Prompt feedback
# if hasattr(response, 'prompt_feedback'):
#     print("\nPrompt Feedback:")
#     print(response.prompt_feedback)
#     print(response.prompt_feedback.block_reason)
#     print(response.prompt_feedback.block_reason.name)

if response.candidates:
    print("candidates", response.candidates)
    print("candidate", response.candidates[0])
    print("candidate content", response.candidates[0].content)
    print("candidate parts", response.candidates[0].content.parts)
    print("candidate content role", response.candidates[0].content.role)
    print("candidate finish reason", response.candidates[0].finish_reason)

# if response.usage_metadata:
#     print("meta data", response.usage_metadata)
#     print(response.usage_metadata.candidates_token_count)
#     print(response.usage_metadata.prompt_token_count)