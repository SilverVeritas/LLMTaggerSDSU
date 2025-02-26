import os
import sys
import json
import logging
from datetime import datetime
import re
import string

import PyPDF2
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("bert_topic_kmeans_nltk")
    logger.setLevel(logging.DEBUG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bert_topic_kmeans_{timestamp}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def extract_text_from_pdf(pdf_path, logger):
    try:
        text_parts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def custom_preprocessor(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def main():
    logger = setup_logging()

    PDF_FOLDER = "test_pdf"
    OUTPUT_JSON = "bert_topic_kmeans_results.json"
    MODEL_NAME = "all-MiniLM-L6-v2"
    N_CLUSTERS = 5

    pdf_files = [p for p in os.listdir(PDF_FOLDER) if p.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files in {PDF_FOLDER}")

    docs = []
    filenames = []
    for pdf_name in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_name)
        text = extract_text_from_pdf(path, logger)
        if text.strip():
            docs.append(text)
            filenames.append(pdf_name)

    if not docs:
        logger.info("No PDF text to process.")
        return

    # Load the SentenceTransformer model
    logger.info(f"Loading sentence-transformer model: {MODEL_NAME}")
    embed_model = SentenceTransformer(MODEL_NAME)
    logger.info("Encoding documents...")
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    # Define NLTK + custom stopwords; convert to a list (important!)
    nltk_stopwords = set(stopwords.words('english'))
    custom_words = {"use", "used", "using", "paper", "however"}
    all_stops = nltk_stopwords.union(custom_words)
    all_stops_list = list(all_stops)  # <-- convert set to list

    # Pass stop_words=all_stops_list (not a set)
    vectorizer_model = CountVectorizer(
        preprocessor=custom_preprocessor,
        stop_words=all_stops_list,
        token_pattern=r"(?u)\b\w\w+\b"
    )

    # Use K-Means for a fixed number of clusters
    kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    topic_model = BERTopic(
        umap_model=None,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    logger.info(f"Fitting BERTopic + K-Means with {N_CLUSTERS} clusters...")
    topics, _ = topic_model.fit_transform(docs, embeddings)

    cluster_map = {}
    for i, topic_id in enumerate(topics):
        cluster_map.setdefault(topic_id, []).append(filenames[i])

    custom_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ")
    output_data = {}
    for topic_id in sorted(cluster_map.keys()):
        if topic_id < len(custom_labels):
            topic_label = f"Topic {topic_id}: {custom_labels[topic_id]}"
        else:
            topic_label = f"Topic {topic_id}"

        top_words_info = topic_model.get_topic(topic_id)[:5] if topic_id >= 0 else []
        top_words = [w[0] for w in top_words_info]

        output_data[topic_label] = {
            "papers": cluster_map[topic_id],
            "top_words": top_words
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
