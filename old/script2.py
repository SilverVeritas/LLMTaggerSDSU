import os
import json
import re
import logging
from datetime import datetime
import sys

# For the openai>=1.0.0 library and dotenv
from openai import OpenAI
from dotenv import load_dotenv

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('paper_clustering')
    logger.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'paper_clustering_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_llm_response(prompt: str, logger: logging.Logger, model: str):
    """
    Calls the OpenAI ChatCompletion endpoint (similar to your original approach).
    Returns the LLM's response content or an error dictionary if failed.
    """

    # Load environment variables (e.g., OPENAI_API_KEY) from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Make sure it's set in .env as OPENAI_API_KEY.")
        return {"error": "OpenAI API key not found."}

    # Create a new OpenAI client
    client = OpenAI(api_key=api_key)

    logger.info("Sending request to LLM for paper clustering...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        llm_output = response.choices[0].message.content

        logger.info("LLM response received.")
        logger.debug(f"Raw LLM response:\n{llm_output}\n")
        return {"response": llm_output}

    except Exception as e:
        logger.error(f"Failed to get LLM response: {str(e)}")
        return {"error": str(e)}

def parse_llm_json(llm_output: str, logger: logging.Logger):
    """
    Strip away any triple backticks or markdown fences from the LLM output,
    then parse as JSON. If it fails to parse, log an error and return an empty dict.
    """
    try:
        # Remove possible ```json ... ``` fences
        cleaned_str = re.sub(r'^```(json)?|```$', '', llm_output.strip(), flags=re.MULTILINE).strip()
        
        # Now parse
        final_dict = json.loads(cleaned_str)
        logger.info("Successfully parsed LLM output as JSON.")
        return final_dict

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM output as JSON: {e}")
        return {}  # Return an empty dict or handle however you see fit

def generate_labels_from_extracted_keywords(
    input_json_path: str,
    output_json_path: str,
    logger: logging.Logger,
    model: str
):
    """
    Reads the 'extracted_keywords.json', calls the LLM to group the papers by labels,
    and saves the result in 'generated_labels.json' (without any 'raw_output' key).
    """

    # Step 1: Load the extracted keywords JSON
    logger.info(f"Loading extracted keywords from {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            extracted_keywords = json.load(f)
    except Exception as e:
        logger.error(f"Could not read input JSON: {e}")
        raise

    # Build a summarizing prompt
    data_for_prompt = []
    for paper_name, value in extracted_keywords.items():
        # The tags from the LLM (or partial data)
        tags = value.get("response", "")
        data_for_prompt.append(f"Paper Name: {paper_name}\nTags:\n{tags}\n")

    joined_papers_text = "\n".join(data_for_prompt)

    # The instruction prompt
    prompt = f"""
We have the following papers, each with extracted tags. Please group them together by generating labels that reflect common themes.
Output a valid JSON object where each key is a label and each value is a list of the paper filenames under that label.
Papers must be in at least 1 label but each paper can be part of multiple labels if it fits.
Do not add extra commentary; only output valid JSON.

Papers and their tags:
{joined_papers_text}

Return a JSON of the form:
{{
  "label1": ["PaperA.pdf", "PaperB.pdf"],
  "label2": ["PaperC.pdf"]
}}
    """

    # Step 2: Send the prompt to the LLM
    response_data = get_llm_response(prompt, logger, model)
    if "error" in response_data:
        logger.error("LLM returned an error; cannot proceed.")
        return

    llm_output = response_data["response"]

    # Step 3: Parse the LLM JSON (remove triple backticks and load)
    final_labels = parse_llm_json(llm_output, logger)

    # Step 4: Save the final dictionary to JSON
    logger.info(f"Saving grouped labels to {output_json_path}")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_labels, out_f, indent=2, ensure_ascii=False)
        logger.info("Labels successfully saved.")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")

def main():
    logger = setup_logging()
    logger.info("Starting paper grouping script.")
    
    # Inputs/outputs
    INPUT_JSON_PATH = "extracted_keywords.json"      # or your actual file
    OUTPUT_JSON_PATH = "generated_labels.json"       # or your desired output
    MODEL = "gpt-4o"                                 # or whichever model you'd like

    try:
        generate_labels_from_extracted_keywords(
            input_json_path=INPUT_JSON_PATH,
            output_json_path=OUTPUT_JSON_PATH,
            logger=logger,
            model=MODEL
        )
        logger.info("Paper grouping process completed.")
    except Exception as e:
        logger.error(f"Paper grouping process failed: {e}")

if __name__ == "__main__":
    main()
