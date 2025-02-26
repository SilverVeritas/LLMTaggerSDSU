import os
from pathlib import Path
import PyPDF2
import json
from typing import Dict, Any
import logging
from datetime import datetime
import sys

from openai import OpenAI
from dotenv import load_dotenv

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('pdf_extractor')
    logger.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'pdf_extraction_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def extract_text_from_pdf(pdf_path: str, logger: logging.Logger) -> str:
    logger.info(f"Starting text extraction from PDF: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(reader.pages, 1):
                logger.debug(f"Processing page {i}/{total_pages}")
                text += page.extract_text() + '\n'
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise

def extract_text_from_txt(txt_path: str, logger: logging.Logger) -> str:
    logger.info(f"Starting text extraction from TXT: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted {len(text)} characters from TXT file")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from TXT file: {str(e)}")
        raise

def get_llm_response(text: str, logger: logging.Logger, model: str) -> Any:
    """
    Calls the OpenAI (1.0.0+) ChatCompletion endpoint WITHOUT streaming.
    We print the entire LLM output after completion, and return it.
    """

    # Load environment variables (e.g., OPENAI_API_KEY) from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Make sure it's set in .env as OPENAI_API_KEY.")
        return {"error": "OpenAI API key not found."}

    # Create a new OpenAI client
    client = OpenAI(api_key=api_key)

    logger.info("Getting LLM response")

    # Example prompt: Adjust as needed for your use case
    prompt = (
        "Extract the most significant keywords and phrases from the following text.\n\n"
        f"{text}\n\n"
        "Return them in a list format without numbering or dashes, instead just separate them with a new line. Do not produce extra commentary."
    )

    try:
        # Create a chat completion (no streaming)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Get the LLM's full response
        llm_output = response.choices[0].message.content

        # Print the entire output
        print("----- LLM RESPONSE START -----")
        print(llm_output)
        print("----- LLM RESPONSE END -----\n")

        return {"response": llm_output}

    except Exception as e:
        logger.error(f"Failed to get LLM response: {str(e)}")
        return {"error": str(e)}

def process_folder(folder_path: str, logger: logging.Logger, model: str) -> Dict[str, Any]:
    logger.info(f"Starting folder processing: {folder_path}")
    results = {}
    folder = Path(folder_path)
    
    # Collect both PDF and TXT files in the folder
    files = list(folder.glob('*.pdf')) + list(folder.glob('*.txt'))
    logger.info(f"Found {len(files)} files (PDF or TXT) to process.")
    
    for file_path in files:
        logger.info(f"\nProcessing file: {file_path.name}")
        try:
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(str(file_path), logger)
            else:
                text = extract_text_from_txt(str(file_path), logger)
                
            response_data = get_llm_response(text, logger, model)
            results[file_path.name] = response_data
            logger.info(f"Completed processing {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            results[file_path.name] = {"error": str(e)}
    
    return results

def save_results(results: Dict[str, Any], output_file: str, logger: logging.Logger):
    logger.info(f"Saving results to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Successfully saved results")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def main():
    logger = setup_logging()
    logger.info("Starting PDF/TXT extraction process (no streaming)")

    FOLDER_PATH = './test_pdf'  
    OUTPUT_FILE = 'extracted_keywords.json'
    MODEL = 'gpt-4o'  
    
    try:
        results = process_folder(FOLDER_PATH, logger, MODEL)
        save_results(results, OUTPUT_FILE, logger)
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
