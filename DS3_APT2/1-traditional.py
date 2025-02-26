import os
import sys
import json
import logging
from datetime import datetime
import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Set

import PyPDF2
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords', download_dir='nltk', quiet=True)
nltk.data.path.append('nltk')
from nltk.corpus import stopwords

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("multilabel_bert_topic")
    logger.setLevel(logging.DEBUG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"multilabel_bert_topic_{timestamp}.log")
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

def convert_to_multilabel_gmm(model, doc_embeddings, docs, topic_to_docs, similarity_threshold=0.2, max_labels=3):
    """
    Post-processes a trained BERTopic model to add multi-label assignments using GMM
    """
    # Extract topic centers from the model
    topic_centers = {}
    unique_topics = set(model.topics_)
    
    # For each topic, calculate its center
    for topic_id in unique_topics:
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get indices of documents in this topic
        doc_indices = [i for i, t in enumerate(model.topics_) if t == topic_id]
        
        if doc_indices:
            # Calculate the mean embedding of all documents in this topic
            topic_center = np.mean(doc_embeddings[doc_indices], axis=0)
            topic_centers[topic_id] = topic_center
    
    # Train a GMM on the document embeddings
    n_components = min(len(topic_centers), 10)  # Limit number of components
    gmm = GaussianMixture(n_components=n_components, random_state=0, covariance_type='full')
    gmm.fit(doc_embeddings)
    
    # Get probabilities for each document
    probs = gmm.predict_proba(doc_embeddings)
    
    # Create multi-label assignments
    multilabel_assignments = {}
    multilabel_topic_to_docs = {k: [] for k in topic_centers.keys()}
    
    for i, (doc_embedding, primary_topic) in enumerate(zip(doc_embeddings, model.topics_)):
        # Skip documents in the outlier topic
        if primary_topic == -1:
            continue
            
        # Calculate similarity to all topic centers
        similarities = {}
        for topic_id, center in topic_centers.items():
            sim = cosine_similarity([doc_embedding], [center])[0][0]
            similarities[topic_id] = sim
        
        # Sort topics by similarity
        sorted_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Assign multiple topics based on similarity threshold
        assigned_topics = []
        for topic_id, sim in sorted_topics:
            if sim >= similarity_threshold:
                assigned_topics.append(topic_id)
                multilabel_topic_to_docs[topic_id].append(i)
                if len(assigned_topics) >= max_labels:
                    break
        
        # Ensure at least one topic is assigned
        if not assigned_topics and primary_topic >= 0:
            assigned_topics = [primary_topic]
            multilabel_topic_to_docs[primary_topic].append(i)
            
        multilabel_assignments[i] = assigned_topics
    
    return multilabel_assignments, multilabel_topic_to_docs

def main():
    logger = setup_logging()

    PDF_FOLDER = r"DS3_APT2\APT"
    OUTPUT_JSON = r"DS3_APT2\trad-1-multilabel_bertopic_results.json"
    MODEL_NAME = "all-MiniLM-L6-v2"
    N_CLUSTERS = 5
    SIMILARITY_THRESHOLD = 0.3  # Threshold for additional topic assignments
    MAX_LABELS = 3              # Maximum number of topics per document

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
        logger.error("No PDF text to process.")
        return

    # Load the SentenceTransformer model
    logger.info(f"Loading sentence-transformer model: {MODEL_NAME}")
    embed_model = SentenceTransformer(MODEL_NAME)
    logger.info("Encoding documents...")
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    # Define NLTK + custom stopwords
    nltk_stopwords = set(stopwords.words('english'))
    custom_words = {"use", "used", "using", "paper", "however"}
    all_stops = nltk_stopwords.union(custom_words)
    all_stops_list = list(all_stops)

    # Configure the vectorizer
    vectorizer_model = CountVectorizer(
        preprocessor=custom_preprocessor,
        stop_words=all_stops_list,
        token_pattern=r"(?u)\b\w\w+\b"
    )

    # Step 1: Run standard BERTopic with KMeans first (this works reliably)
    kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    
    topic_model = BERTopic(
        umap_model=None,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )

    logger.info(f"Fitting BERTopic with KMeans ({N_CLUSTERS} clusters)")
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # Create standard topic mapping
    topic_to_docs = {}
    for i, topic_id in enumerate(topics):
        if topic_id not in topic_to_docs:
            topic_to_docs[topic_id] = []
        topic_to_docs[topic_id].append(i)
    
    # Step 2: Convert to multi-label assignments
    logger.info("Creating multi-label assignments using document similarities")
    multilabel_assignments, multilabel_topic_to_docs = convert_to_multilabel_gmm(
        topic_model, 
        embeddings, 
        docs,
        topic_to_docs,
        similarity_threshold=SIMILARITY_THRESHOLD,
        max_labels=MAX_LABELS
    )
    
    # Calculate average topics per document
    avg_topics = sum(len(topics) for topics in multilabel_assignments.values()) / len(multilabel_assignments)
    logger.info(f"Average topics per document: {avg_topics:.2f}")
    
    # Build the final output structure
    output_data = {}
    
    for topic_id in sorted(multilabel_topic_to_docs.keys()):
        if topic_id >= 0:  # Skip outlier topic
            # Get topic label
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                label_words = ", ".join([word for word, _ in topic_words[:3]])
                topic_label = f"Topic {topic_id}: {label_words}"
            else:
                topic_label = f"Topic {topic_id}"
                
            # Get document filenames for this topic
            doc_indices = multilabel_topic_to_docs[topic_id]
            doc_names = [filenames[i] for i in doc_indices]
            
            # Get top words
            top_words = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
            
            # Add to output
            output_data[topic_label] = {
                "papers": doc_names,
                "top_words": top_words
            }
    
    # Save results to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved multi-label results to {OUTPUT_JSON}")
    
    # Save metadata
    metadata = {
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "max_labels": MAX_LABELS,
        "avg_topics_per_doc": float(avg_topics),
        "total_docs": len(docs),
        "total_topics": len(output_data)
    }
    
    metadata_file = OUTPUT_JSON.replace(".json", "_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

if __name__ == "__main__":
    main()