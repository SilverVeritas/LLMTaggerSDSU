import os
import sys
import json
import re
import logging
from datetime import datetime
import PyPDF2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import Counter

# For the OpenAI API
from openai import OpenAI
from dotenv import load_dotenv

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('document_labeling')
    logger.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'document_labeling_{timestamp}.log')
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
    """Extract text from a PDF file."""
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
    """Extract text from a TXT file."""
    logger.info(f"Starting text extraction from TXT: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted {len(text)} characters from TXT file")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from TXT file: {str(e)}")
        raise

def get_llm_response(prompt: str, logger: logging.Logger, model: str) -> Dict[str, Any]:
    """Call the OpenAI API with the given prompt."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Make sure it's set in .env as OPENAI_API_KEY.")
        return {"error": "OpenAI API key not found."}

    # Create a new OpenAI client
    client = OpenAI(api_key=api_key)

    logger.info("Getting LLM response")

    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Get the LLM's full response
        llm_output = response.choices[0].message.content

        # Print the output
        print("----- LLM RESPONSE START -----")
        print(llm_output)
        print("----- LLM RESPONSE END -----\n")

        return {"response": llm_output}

    except Exception as e:
        logger.error(f"Failed to get LLM response: {str(e)}")
        return {"error": str(e)}

def summarize_text(text: str, logger: logging.Logger, model: str) -> str:
    """Summarize document text using an LLM."""
    logger.info("Summarizing document text")
    
    # Truncate very long texts to fit within context window
    max_chars = 99999 # not truncating for now
    if len(text) > max_chars:
        logger.info(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
        text = text[:max_chars] + "..."
    
    # Create a prompt for summarization
    prompt = (
        "Summarize the following document text concisely. Preserve the key concepts, arguments, and findings. "
        "Focus on the main topic and important details that would be relevant for categorizing this document.\n\n"
        f"{text}\n\n"
        "Summary:"
    )
    
    # Get summary from LLM
    response = get_llm_response(prompt, logger, model)
    
    if "error" in response:
        logger.error("Failed to summarize text")
        return ""
    
    summary = response["response"]
    logger.info(f"Successfully generated summary of {len(summary)} characters")
    return summary

def extract_keywords(text: str, logger: logging.Logger, model: str) -> List[str]:
    """Extract keywords from text using an LLM."""
    logger.info("Extracting keywords from text")
    
    # Create a prompt for keyword extraction
    prompt = (
        "Extract the most significant keywords and phrases from the following text.\n\n"
        f"{text}\n\n"
        "Return them in a list format without numbering or dashes, instead just separate them with a new line. Do not produce extra commentary."
    )
    
    # Get keywords from LLM
    response = get_llm_response(prompt, logger, model)
    
    if "error" in response:
        logger.error("Failed to extract keywords")
        return []
    
    # Parse the response into a list of keywords
    keywords = [kw.strip() for kw in response["response"].split('\n') if kw.strip()]
    logger.info(f"Successfully extracted {len(keywords)} keywords")
    return keywords

def generate_labels(docs_keywords: Dict[str, List[str]], logger: logging.Logger, model: str) -> Dict[str, List[str]]:
    """Generate topic labels from document keywords using an LLM."""
    logger.info("Generating topic labels from keywords")
    
    # Prepare data for prompt
    data_for_prompt = []
    for doc_name, keywords in docs_keywords.items():
        keywords_text = '\n'.join(keywords)
        data_for_prompt.append(f"Document: {doc_name}\nKeywords:\n{keywords_text}\n")
    
    joined_docs_text = "\n".join(data_for_prompt)
    
    # Create prompt for label generation
    prompt = f"""
We have the following documents with their extracted keywords. Please group them by generating labels that reflect common themes.
Output a valid JSON object where each key is a label and each value is a list of the document filenames.
Documents must be in at least 1 label but each document can be part of multiple labels if it fits.
Do not add extra commentary; only output valid JSON.

Documents and their keywords:
{joined_docs_text}

Return a JSON of the form:
{{
  "label1": ["Document1.pdf", "Document2.pdf"],
  "label2": ["Document3.pdf"]
}}
    """
    
    # Get labels from LLM
    response = get_llm_response(prompt, logger, model)
    
    if "error" in response:
        logger.error("Failed to generate labels")
        return {}
    
    # Parse the JSON response
    labels_dict = parse_llm_json(response["response"], logger)
    logger.info(f"Successfully generated {len(labels_dict)} labels")
    return labels_dict

def parse_llm_json(llm_output: str, logger: logging.Logger) -> Dict:
    """Parse JSON from LLM output, handling markdown code blocks."""
    try:
        # Remove possible ```json ... ``` fences
        cleaned_str = re.sub(r'^```(json)?|```$', '', llm_output.strip(), flags=re.MULTILINE).strip()
        
        # Parse JSON
        final_dict = json.loads(cleaned_str)
        logger.info("Successfully parsed LLM output as JSON.")
        return final_dict

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM output as JSON: {e}")
        return {}

def vectorize_texts(papers_text: Dict[str, str], logger: logging.Logger):
    """Convert document texts to TF-IDF vectors."""
    doc_names = list(papers_text.keys())
    texts = [papers_text[n] for n in doc_names]
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(texts)
    logger.info(f"TF–IDF matrix shape: {X.shape}")
    return doc_names, X

def compute_multilabel_density(labels_dict, X, doc_names, logger: logging.Logger):
    """Compute density-based metrics for each label."""
    logger.info("Computing multi-label density metrics...")
    
    # Create document name to index mapping
    doc_to_idx = {name: i for i, name in enumerate(doc_names) if i < X.shape[0]}
    
    results = {}
    for lbl, docs in labels_dict.items():
        # Get indices for docs in this label
        valid_indices = [doc_to_idx[doc] for doc in docs if doc in doc_to_idx]
        
        if len(valid_indices) > 1:
            # Extract the vectors for these documents
            vectors = X[valid_indices]
            
            # Compute pairwise cosine similarities
            sim_matrix = 1 - pairwise_distances(vectors, metric='cosine')
            
            # Average similarity (density) - only upper triangle without diagonal
            n = sim_matrix.shape[0]
            sum_sim = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    sum_sim += sim_matrix[i, j]
                    count += 1
            
            density = float(sum_sim / count) if count > 0 else 0.0
            
            results[lbl] = density
            logger.debug(f"Label '{lbl}' density: {density:.4f}")
        else:
            results[lbl] = 0.0
            logger.debug(f"Label '{lbl}' has fewer than 2 valid documents, density set to 0")
    
    # Overall average density
    overall = np.mean(list(results.values())) if results else 0
    logger.info(f"Overall average density: {overall:.4f}")
    
    return results, overall

def compute_multiple_coherence_metrics(labels_dict, papers_text, logger: logging.Logger):
    """Compute multiple coherence metrics for topic evaluation."""
    logger.info("Computing multiple coherence metrics...")
    
    tokenizer = re.compile(r"\w+")
    label_tokens = {}
    
    for lbl, papers in labels_dict.items():
        all_toks = []
        for p in papers:
            txt = papers_text.get(p, "")
            all_toks += tokenizer.findall(txt.lower())
        label_tokens[lbl] = all_toks
        logger.debug(f"Label '{lbl}': extracted {len(all_toks)} tokens")
    
    dct = Dictionary(label_tokens.values())
    logger.debug(f"Created dictionary with {len(dct)} unique tokens")
    
    bow_map = {lbl: dct.doc2bow(toks) for lbl, toks in label_tokens.items()}
    
    # Extract top words per topic
    tops = []
    top_words_per_label = {}
    for lbl, bow in bow_map.items():
        sorted_bow = sorted(bow, key=lambda x: x[1], reverse=True)[:20]
        top_words = [dct[w] for (w, _) in sorted_bow]
        tops.append(top_words)
        top_words_per_label[lbl] = top_words
        logger.debug(f"Label '{lbl}' top words: {', '.join(top_words[:5])}...")
    
    # Compute different coherence measures
    coherence_metrics = {}
    for coherence_type in ['c_v', 'c_uci', 'c_npmi']:
        try:
            cm = CoherenceModel(
                topics=tops,
                texts=list(label_tokens.values()),
                dictionary=dct,
                coherence=coherence_type
            )
            coherence_val = cm.get_coherence()
            coherence_metrics[coherence_type] = float(coherence_val)
            logger.info(f"Coherence ({coherence_type}): {coherence_val:.4f}")
        except Exception as e:
            logger.error(f"Error computing {coherence_type} coherence: {e}")
            coherence_metrics[coherence_type] = None
    
    return coherence_metrics, top_words_per_label

def compute_overlap_quality(labels_dict, X, doc_names, logger: logging.Logger):
    """Measure how well the clustering handles overlapping assignments."""
    logger.info("Computing overlap quality metrics...")
    
    # Create document name to index mapping
    doc_to_idx = {name: i for i, name in enumerate(doc_names) if i < X.shape[0]}
    
    # Create doc_to_labels mapping
    doc_to_labels = {}
    for label, docs in labels_dict.items():
        for doc in docs:
            if doc not in doc_to_labels:
                doc_to_labels[doc] = []
            doc_to_labels[doc].append(label)
    
    # Calculate average number of labels per document
    avg_labels_per_doc = sum(len(labels) for labels in doc_to_labels.values()) / len(doc_to_labels) if doc_to_labels else 0
    logger.info(f"Average labels per document: {avg_labels_per_doc:.2f}")
    
    # Pre-compute full distance matrix once
    logger.debug("Computing full distance matrix...")
    distance_matrix = pairwise_distances(X, metric='cosine')
    similarity_matrix = 1.0 - distance_matrix
    
    # Calculate similarity between documents that share labels vs those that don't
    shared_label_similarities = []
    diff_label_similarities = []
    
    # Get the list of documents that have labels and exist in our vector space
    valid_docs = [doc for doc in doc_to_labels if doc in doc_to_idx]
    
    for i in range(len(valid_docs)):
        doc1 = valid_docs[i]
        idx1 = doc_to_idx[doc1]
        labels1 = set(doc_to_labels[doc1])
        
        for j in range(i+1, len(valid_docs)):
            doc2 = valid_docs[j]
            idx2 = doc_to_idx[doc2]
            labels2 = set(doc_to_labels[doc2])
            
            # Get the similarity from our pre-computed matrix
            sim = similarity_matrix[idx1, idx2]
            
            # Check if they share any labels
            if labels1.intersection(labels2):
                shared_label_similarities.append(sim)
            else:
                diff_label_similarities.append(sim)
    
    # Calculate average similarities
    avg_shared_sim = sum(shared_label_similarities) / len(shared_label_similarities) if shared_label_similarities else 0
    avg_diff_sim = sum(diff_label_similarities) / len(diff_label_similarities) if diff_label_similarities else 0
    
    logger.info(f"Average similarity between docs sharing labels: {avg_shared_sim:.4f}")
    logger.info(f"Average similarity between docs with different labels: {avg_diff_sim:.4f}")
    
    # Quality index - higher is better
    overlap_quality = avg_shared_sim - avg_diff_sim
    logger.info(f"Overlap quality index: {overlap_quality:.4f}")
    
    return {
        "avg_labels_per_doc": float(avg_labels_per_doc),
        "avg_shared_label_similarity": float(avg_shared_sim),
        "avg_diff_label_similarity": float(avg_diff_sim),
        "overlap_quality_index": float(overlap_quality),
        "shared_pairs_count": len(shared_label_similarities),
        "diff_pairs_count": len(diff_label_similarities)
    }

def compute_topic_distinctiveness(labels_dict, papers_text, logger: logging.Logger):
    """Measure how distinct each topic/label is from others."""
    logger.info("Computing topic distinctiveness...")
    
    # Extract top words for each label
    tokenizer = re.compile(r"\w+")
    label_word_freq = {}
    
    for lbl, papers in labels_dict.items():
        word_count = {}
        total_words = 0
        
        for p in papers:
            txt = papers_text.get(p, "")
            words = tokenizer.findall(txt.lower())
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
                total_words += 1
        
        # Normalize by total words
        if total_words > 0:
            label_word_freq[lbl] = {word: count/total_words for word, count in word_count.items()}
            logger.debug(f"Label '{lbl}': processed {total_words} words")
        else:
            label_word_freq[lbl] = {}
            logger.warning(f"Label '{lbl}': no words found")
    
    # Compute Jensen-Shannon divergence between label distributions
    distinctiveness_scores = {}
    
    for lbl1 in label_word_freq:
        js_divs = []
        for lbl2 in label_word_freq:
            if lbl1 != lbl2:
                # Create probability distributions
                all_words = set(label_word_freq[lbl1].keys()) | set(label_word_freq[lbl2].keys())
                p = np.array([label_word_freq[lbl1].get(word, 0) for word in all_words])
                q = np.array([label_word_freq[lbl2].get(word, 0) for word in all_words])
                
                # Normalize if needed
                if np.sum(p) > 0:
                    p = p / np.sum(p)
                if np.sum(q) > 0:
                    q = q / np.sum(q)
                
                # Compute JS divergence
                m = 0.5 * (p + q)
                # Avoid log(0) by adding small epsilon
                js_div = 0.5 * np.sum(p * np.log2(p/m + 1e-10)) + 0.5 * np.sum(q * np.log2(q/m + 1e-10))
                js_divs.append(js_div)
        
        distinctiveness_scores[lbl] = float(np.mean(js_divs)) if js_divs else 0
        logger.debug(f"Label '{lbl}' distinctiveness: {distinctiveness_scores[lbl]:.4f}")
    
    overall_distinctiveness = float(np.mean(list(distinctiveness_scores.values()))) if distinctiveness_scores else 0
    logger.info(f"Overall topic distinctiveness: {overall_distinctiveness:.4f}")
    
    return distinctiveness_scores, overall_distinctiveness

def compute_label_balance(labels_dict, logger: logging.Logger):
    """Measure how balanced the label assignments are."""
    logger.info("Computing label balance metrics...")
    
    # Count documents per label
    label_sizes = {label: len(docs) for label, docs in labels_dict.items()}
    logger.debug(f"Label sizes: {label_sizes}")
    
    # Count unique documents
    all_docs = set()
    for docs in labels_dict.values():
        all_docs.update(docs)
    
    # Count labels per document
    doc_label_counts = {}
    for label, docs in labels_dict.items():
        for doc in docs:
            doc_label_counts[doc] = doc_label_counts.get(doc, 0) + 1
    
    # Calculate statistics
    label_counts = list(label_sizes.values())
    label_size_stats = {
        "min": int(min(label_counts)) if label_counts else 0,
        "max": int(max(label_counts)) if label_counts else 0,
        "mean": float(np.mean(label_counts)) if label_counts else 0,
        "std": float(np.std(label_counts)) if label_counts else 0
    }
    
    doc_counts = list(doc_label_counts.values())
    doc_label_stats = {
        "min": int(min(doc_counts)) if doc_counts else 0,
        "max": int(max(doc_counts)) if doc_counts else 0,
        "mean": float(np.mean(doc_counts)) if doc_counts else 0,
        "std": float(np.std(doc_counts)) if doc_counts else 0
    }
    
    # Calculate entropy as measure of balance
    total_docs_in_labels = sum(label_sizes.values())
    if total_docs_in_labels > 0:
        label_probs = [size/total_docs_in_labels for size in label_sizes.values()]
        label_entropy = float(-np.sum(p * np.log2(p) for p in label_probs if p > 0))
    else:
        label_entropy = 0.0
    
    logger.info(f"Total labels: {len(labels_dict)}")
    logger.info(f"Total unique documents: {len(all_docs)}")
    logger.info(f"Label distribution entropy: {label_entropy:.4f}")
    logger.info(f"Average labels per document: {doc_label_stats['mean']:.2f}")
    
    return {
        "label_size_stats": label_size_stats,
        "doc_label_stats": doc_label_stats,
        "label_entropy": label_entropy,
        "total_labels": len(labels_dict),
        "total_docs": len(all_docs)
    }

def process_documents_pipeline(
    folder_path: str,
    summaries_json_path: str,
    keywords_json_path: str,
    labels_json_path: str,
    eval_json_path: str,
    logger: logging.Logger,
    model: str
):
    """Process documents through the complete pipeline: summarize → extract → label → evaluate."""
    logger.info("Starting document processing pipeline")
    
    # Step 1: Get document texts
    folder = Path(folder_path)
    files = list(folder.glob('*.pdf')) + list(folder.glob('*.txt'))
    logger.info(f"Found {len(files)} files (PDF or TXT) to process")
    
    doc_texts = {}
    for file_path in files:
        logger.info(f"Processing file: {file_path.name}")
        try:
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(str(file_path), logger)
            else:
                text = extract_text_from_txt(str(file_path), logger)
                
            doc_texts[file_path.name] = text
            logger.info(f"Completed text extraction for {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    # Step 2: Summarize documents
    logger.info("Starting document summarization")
    doc_summaries = {}
    for doc_name, text in doc_texts.items():
        logger.info(f"Summarizing {doc_name}")
        summary = summarize_text(text, logger, model)
        doc_summaries[doc_name] = summary
    
    # Save summaries
    logger.info(f"Saving document summaries to {summaries_json_path}")
    with open(summaries_json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_summaries, f, indent=2, ensure_ascii=False)
    
    # Step 3: Extract keywords from summaries
    logger.info("Starting keyword extraction from summaries")
    doc_keywords = {}
    for doc_name, summary in doc_summaries.items():
        logger.info(f"Extracting keywords from {doc_name}")
        keywords = extract_keywords(summary, logger, model)
        doc_keywords[doc_name] = keywords
    
    # Save keywords
    logger.info(f"Saving document keywords to {keywords_json_path}")
    with open(keywords_json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_keywords, f, indent=2, ensure_ascii=False)
    
    # Step 4: Generate labels from keywords
    logger.info("Generating labels from keywords")
    labels_dict = generate_labels(doc_keywords, logger, model)
    
    # Save labels
    logger.info(f"Saving generated labels to {labels_json_path}")
    with open(labels_json_path, 'w', encoding='utf-8') as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)
    
    # Step 5: Evaluate labels
    logger.info("Evaluating generated labels")
    doc_names, X = vectorize_texts(doc_texts, logger)
    
    # Compute evaluation metrics
    results = {}
    
    # 1. Compute multi-label density metrics
    label_densities, overall_density = compute_multilabel_density(labels_dict, X, doc_names, logger)
    results["label_densities"] = label_densities
    results["overall_density"] = overall_density
    
    # 2. Compute coherence metrics
    coherence_metrics, top_words = compute_multiple_coherence_metrics(labels_dict, doc_texts, logger)
    results["coherence_metrics"] = coherence_metrics
    
    # Store top words per label (useful for interpretation)
    results["top_words_per_label"] = {lbl: words for lbl, words in top_words.items()}
    
    # 3. Compute overlap quality metrics
    overlap_metrics = compute_overlap_quality(labels_dict, X, doc_names, logger)
    results["overlap_metrics"] = overlap_metrics
    
    # 4. Compute topic distinctiveness
    topic_distinctiveness, overall_distinctiveness = compute_topic_distinctiveness(labels_dict, doc_texts, logger)
    results["topic_distinctiveness"] = topic_distinctiveness
    results["overall_distinctiveness"] = overall_distinctiveness
    
    # 5. Compute label balance metrics
    balance_metrics = compute_label_balance(labels_dict, logger)
    results["balance_metrics"] = balance_metrics
    
    # Save evaluation results
    logger.info(f"Saving evaluation results to {eval_json_path}")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print a summary of key metrics
    logger.info("\n==== SUMMARY OF KEY METRICS ====")
    logger.info(f"Overall Label Density: {overall_density:.4f}")
    if coherence_metrics.get("c_v") is not None:
        logger.info(f"Topic Coherence (C_V): {coherence_metrics['c_v']:.4f}")
    logger.info(f"Overlap Quality Index: {overlap_metrics['overlap_quality_index']:.4f}")
    logger.info(f"Topic Distinctiveness: {overall_distinctiveness:.4f}")
    logger.info(f"Label Entropy: {balance_metrics['label_entropy']:.4f}")
    logger.info(f"Avg Labels per Doc: {balance_metrics['doc_label_stats']['mean']:.2f}")
    logger.info("==============================")
    
    logger.info("Document processing pipeline completed successfully")
    
    return results

def main():
    # Paths and config
    FOLDER_PATH = r'DS3_APT2\APT'
    SUMMARIES_JSON_PATH = r'DS3_APT2\summary_documents.json'
    KEYWORDS_JSON_PATH = r'DS3_APT2\summary_extracted_keywords.json'
    LABELS_JSON_PATH = r'DS3_APT2\summary_generated_labels.json'
    EVAL_JSON_PATH = r'DS3_APT2\summary_evaluation_results.json'
    MODEL = 'gpt-4o'  
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting enhanced document processing pipeline with summarization")
    
    try:
        # Run the complete pipeline
        process_documents_pipeline(
            folder_path=FOLDER_PATH,
            summaries_json_path=SUMMARIES_JSON_PATH,
            keywords_json_path=KEYWORDS_JSON_PATH,
            labels_json_path=LABELS_JSON_PATH,
            eval_json_path=EVAL_JSON_PATH,
            logger=logger,
            model=MODEL
        )
        
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()