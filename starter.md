Welcome to Omini Problem Generator!
This guide will walk you through setting up and running the Omini Problem Generator project. This powerful tool uses Google's Gemini AI to automatically generate mathematical questions and answers from image-based inputs, including pages extracted from PDF documents.

By following these steps, you'll be able to:

Set up your development environment.

Securely configure your Google Gemini API key.

Process your own image and PDF files to generate new Q&A datasets.

Let's get started!

Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.8+: Download and install from python.org.

Git: For cloning the repository. Download from git-scm.com.

Poppler: This is an external command-line utility that pdf2image (a Python library used in this project) relies on to convert PDFs to images.

macOS (using Homebrew):

brew install poppler

Linux (Debian/Ubuntu):

sudo apt-get update
sudo apt-get install -y poppler-utils

Windows:

Download a pre-compiled binary distribution of Poppler. A common source is this blog post (look for the latest Poppler for Windows link).

Extract the downloaded zip file to a convenient location (e.g., C:\poppler).

Add the bin subdirectory of the extracted Poppler folder to your system's PATH environment variable. This allows your system to find the Poppler executables. If you're unsure how to do this, a quick search for "add to PATH Windows" for your Windows version will provide detailed instructions.

Setup Guide
Follow these steps to get the project ready to run:

1. Clone the Repository
Open your terminal or command prompt and run the following command to download the project to your local machine:

git clone https://github.com/YOUR_GITHUB_USERNAME/Omini-Problem-Generator.git

Replace YOUR_GITHUB_USERNAME with the actual GitHub username where the repository is hosted.

Now, navigate into the project directory:

cd Omini-Problem-Generator

2. Create and Activate a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies. This isolates the project's libraries from your global Python installation.

python3 -m venv venv

(Use python -m venv venv if python3 is not found or python is your primary Python 3 executable).

Activate your virtual environment:

On macOS / Linux:

source venv/bin/activate

On Windows (Command Prompt):

venv\Scripts\activate.bat

On Windows (PowerShell):

venv\Scripts\Activate.ps1

You will know the virtual environment is active when (venv) appears at the beginning of your terminal prompt.

3. Install Python Dependencies
With your virtual environment activated, install all required Python libraries using the requirements.txt file:

pip install -r requirements.txt

4. Get Your Google Gemini API Key
This project requires access to the Google Gemini API. You'll need to generate an API key and store it securely.

Go to Google AI Studio: Visit https://makersuite.google.com/app/apikey.

Generate an API Key: If you don't have one, click "Create API key in new project" or "Create API key in existing project". Copy your newly generated API key.

Create a .env file: In the root directory of your Omini-Problem-Generator project (the same directory where main.py and requirements.txt are located), create a new file named .env.

You can do this via your terminal:

macOS / Linux: touch .env

Windows: type nul > .env

Or simply create a new file named .env using your text editor.

Add your API Key to .env: Open the .env file and add the following line, replacing YOUR_ACTUAL_GEMINI_API_KEY_HERE with the key you copied from Google AI Studio:

GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"

Important: The .env file is intentionally ignored by Git (see .gitignore). Never share your API key publicly or commit it to your version control system.

5. Prepare Input Data
The project can process either individual image files or multi-page PDF documents.

For Images: Place your .png, .jpg, .jpeg, .gif, or .bmp image files into the data/input_images/ directory.

For PDFs: Place your .pdf documents into the data/input_pdfs/ directory. For example, you might place a mathematical textbook or document here.

You can mix and match; the pipeline allows you to specify which sources to process.

Running the Project
Once all the setup steps are complete, you can run the main pipeline.

Ensure your virtual environment is active (Step 2).

Open main.py in your text editor. You'll see configuration flags here:

# main.py
# ...
    pipeline = MathQnAPipeline(
        input_pdf_folder=PDF_DIR,
        input_image_folder=IMAGE_DIR,
        output_base_dir=OUTPUT_DIR,
        temp_image_output_dir=TEMP_IMAGE_OUTPUT_DIR,
        max_pdf_pages_to_process=1, # Adjust this: e.g., 1 for quick test, None for all pages
        process_images_from_folder=False, # Set to True to process images from data/input_images
        process_pdfs_from_folder=True     # Set to True to process PDFs from data/input_pdfs
    )
# ...

Adjust the flags process_images_from_folder and process_pdfs_from_folder to True based on whether you want to process images, PDFs, or both.

If you set process_pdfs_from_folder=True, you can also adjust max_pdf_pages_to_process to limit the number of pages processed from each PDF (useful for quick tests). Set to None to process all pages.

Run the pipeline from the root of your project directory:

python main.py

The script will print progress messages to your console as it performs API calls and processes data.

Understanding the Output
After the pipeline completes, check the output/ directory in your project root.

output/final_qna_dataset.json: This JSON file will contain a structured list of all generated Q&A pairs, solutions (from LLM1 and LLM2), extracted LaTeX, evaluations, topics, and other metadata for each processed image.

output/final_qna_dataset.csv: This CSV file provides the same data in a tabular format, suitable for spreadsheet analysis. Note that list/dictionary fields (like critical_expressions or topics) will be represented as JSON strings within the CSV cells.

You'll also find a temp_images/ directory. This is where individual page images extracted from your PDF files are temporarily stored during processing. You can safely clear this folder after a successful run if you don't need these intermediate images.

Troubleshooting
ModuleNotFoundError: No module named 'src': Ensure you are running python main.py from the root of your Omini-Problem-Generator directory (where main.py is located), and that your virtual environment is active.

API Key Errors (e.g., "GEMINI_API_KEY not found"): Double-check that you've created the .env file correctly, that it's in the root directory, and that GEMINI_API_KEY="YOUR_KEY" is properly set with your actual key.

PDF Conversion Errors (related to Poppler): If pdf2image fails, it's almost certainly because Poppler is not correctly installed or its bin directory is not in your system's PATH. Revisit the Poppler installation steps for your operating system.

Rate Limit Errors: If you hit API rate limits, the call_gemini_api_with_retries function has built-in back-off. If errors persist, consider reducing the number of max_pdf_pages_to_process or max_retries in the code, or explore increasing your Google Gemini API quota in the Google Cloud Console.

If you encounter any other issues, feel free to ask for help!