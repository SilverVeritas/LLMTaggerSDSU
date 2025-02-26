import os
import sys
import json
import re
import logging
from datetime import datetime
from typing import Dict
from scipy.sparse import vstack


import PyPDF2
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import Counter

def setup_logging(log_dir="logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('label_evaluation')
    logger.setLevel(logging.DEBUG)
    fn = os.path.join(
        log_dir,
        f'label_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    fh = logging.FileHandler(fn)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def extract_text_from_pdf(pdf_path: str, logger: logging.Logger) -> str:
    try:
        text_parts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pg_text = page.extract_text() or ""
                text_parts.append(pg_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"PDF read error {pdf_path}: {e}")
        return ""

def vectorize_texts(papers_text: Dict[str, str], logger: logging.Logger):
    doc_names = list(papers_text.keys())
    texts = [papers_text[n] for n in doc_names]
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(texts)
    logger.info(f"TF–IDF matrix shape: {X.shape}")
    return doc_names, X

def compute_multilabel_density(labels_dict, X, doc_names, logger: logging.Logger):
    """
    Computes density-based metrics for each label
    """
    logger.info("Computing multi-label density metrics...")
    
    # Create document name to index mapping for quick lookup
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
    """
    Computes multiple coherence metrics for topic evaluation
    """
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
    """
    Measures how well the clustering handles overlapping assignments
    """
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
    """
    Measures how distinct each topic/label is from others
    """
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
    """
    Measures how balanced the label assignments are
    """
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

def main():
    LABELS_JSON_PATH = r"DS1_Paper/llm_2_generated_labels.json"
    PDF_FOLDER = r"DS1_Paper/pdfs"
    EVAL_OUTPUT_JSON = r"DS1_Paper/llm_3_evaluation_results.json"

    logger = setup_logging()

    try:
        with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        logger.info(f"Loaded labels from {LABELS_JSON_PATH}")
        logger.info(f"Found {len(labels_dict)} labels")
    except Exception as e:
        logger.error(f"Could not load labels JSON: {e}")
        return

    pdfs = [p for p in os.listdir(PDF_FOLDER) if p.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdfs)} PDF files")
    
    papers_text = {}
    for pdf_name in pdfs:
        path = os.path.join(PDF_FOLDER, pdf_name)
        papers_text[pdf_name] = extract_text_from_pdf(path, logger)
    logger.info(f"Extracted text from {len(papers_text)} PDFs")

    doc_names, X = vectorize_texts(papers_text, logger)
    
    results = {}
    
    # 1. Compute multi-label density metrics
    label_densities, overall_density = compute_multilabel_density(labels_dict, X, doc_names, logger)
    results["label_densities"] = label_densities
    results["overall_density"] = overall_density
    
    # 2. Compute coherence metrics
    coherence_metrics, top_words = compute_multiple_coherence_metrics(labels_dict, papers_text, logger)
    results["coherence_metrics"] = coherence_metrics
    
    # Store top words per label (useful for interpretation)
    results["top_words_per_label"] = {lbl: words for lbl, words in top_words.items()}
    
    # 3. Compute overlap quality metrics
    overlap_metrics = compute_overlap_quality(labels_dict, X, doc_names, logger)
    results["overlap_metrics"] = overlap_metrics
    
    # 4. Compute topic distinctiveness
    topic_distinctiveness, overall_distinctiveness = compute_topic_distinctiveness(labels_dict, papers_text, logger)
    results["topic_distinctiveness"] = topic_distinctiveness
    results["overall_distinctiveness"] = overall_distinctiveness
    
    # 5. Compute label balance metrics
    balance_metrics = compute_label_balance(labels_dict, logger)
    results["balance_metrics"] = balance_metrics
    
    # Save all results to JSON
    logger.info(f"Saving results to {EVAL_OUTPUT_JSON}")
    with open(EVAL_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation complete.")
    
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

if __name__ == "__main__":
    main()