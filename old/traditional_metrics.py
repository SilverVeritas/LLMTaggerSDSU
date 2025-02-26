import os
import sys
import json
import re
import logging
from datetime import datetime
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, pairwise_distances, normalized_mutual_info_score
from sklearn.cluster import KMeans
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def setup_logging(log_dir="logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("metrics_evaluation")
    logger.setLevel(logging.DEBUG)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"metrics_{ts}.log"))
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

def vectorize_texts(papers_text):
    doc_names = list(papers_text.keys())
    texts = [papers_text[n] for n in doc_names]
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(texts)
    return doc_names, X

def compute_silhouette_score_custom(X, labels):
    return silhouette_score(X, labels=labels, metric='cosine')

def compute_intra_label_similarity(X, labels):
    dist = pairwise_distances(X, metric='cosine')
    sim = 1.0 - dist
    label_indices = {}
    for i, lbl in enumerate(labels):
        label_indices.setdefault(lbl, []).append(i)
    sims = {}
    for lbl, idxs in label_indices.items():
        if len(idxs) > 1:
            m = sim[np.ix_(idxs, idxs)]
            tri = np.triu_indices(len(idxs), k=1)
            cluster_sim = np.mean(m[tri])
        else:
            cluster_sim = float('nan')
        sims[lbl] = cluster_sim
    valid = [v for v in sims.values() if not np.isnan(v)]
    overall = np.mean(valid) if valid else 0
    return sims, overall

def compute_topic_coherence(labels_dict, papers_text):
    tokenizer = re.compile(r"\w+")
    label_tokens = {}
    for lbl, pdfs in labels_dict.items():
        all_toks = []
        for p in pdfs:
            txt = papers_text.get(p, "")
            all_toks += tokenizer.findall(txt.lower())
        label_tokens[lbl] = all_toks
    dct = Dictionary(label_tokens.values())
    bow_map = {lbl: dct.doc2bow(toks) for lbl, toks in label_tokens.items()}
    topics_list = []
    for lbl, bow in bow_map.items():
        sorted_bow = sorted(bow, key=lambda x: x[1], reverse=True)[:20]
        topics_list.append([dct[w] for (w, _) in sorted_bow])
    cm = CoherenceModel(
        topics=topics_list,
        texts=list(label_tokens.values()),
        dictionary=dct,
        coherence='c_v'
    )
    return cm.get_coherence()

def compute_nmi(X, labels, n_clusters, logger):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    preds = km.fit_predict(X)
    val = normalized_mutual_info_score(labels, preds)
    logger.info(f"NMI vs KMeans({n_clusters}): {val:.4f}")
    return val

def main():
    logger = setup_logging()

    CLUSTERS_JSON = "bert_topic_kmeans_results.json"
    PDF_FOLDER = "test_pdf"
    EVAL_OUTPUT_JSON = "evaluation_metrics_traditional.json"

    # Load cluster output (Topic 0, 1, etc.)
    try:
        with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
    except Exception as e:
        logger.error(f"Cannot load {CLUSTERS_JSON}: {e}")
        return

    # Parse cluster IDs (e.g. "Topic 0: label") -> numeric 0
    # Build a dict cluster_id -> list of PDFs
    cluster_map = {}
    doc_to_cluster = {}
    cluster_id = 0
    for full_topic_label, info in clusters_data.items():
        # E.g. "Topic 0:" -> 0
        # We'll parse the first integer we see
        match = re.search(r"Topic\s+(\d+)", full_topic_label)
        if match:
            cid = int(match.group(1))
        else:
            cid = cluster_id
        cluster_map[cid] = info["papers"]
        for pdf in info["papers"]:
            doc_to_cluster[pdf] = cid
        cluster_id += 1

    # Extract text from PDF folder
    pdf_files = [p for p in os.listdir(PDF_FOLDER) if p.lower().endswith(".pdf")]
    papers_text = {}
    for pdf_name in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_name)
        text = extract_text_from_pdf(path, logger)
        papers_text[pdf_name] = text

    doc_names, X = vectorize_texts(papers_text)

    assigned_labels = []
    to_remove = []
    for i, dname in enumerate(doc_names):
        if dname in doc_to_cluster:
            assigned_labels.append(doc_to_cluster[dname])
        else:
            to_remove.append(i)
            assigned_labels.append(-1)

    # Remove docs that weren't assigned
    if to_remove:
        for idx in sorted(to_remove, reverse=True):
            X = X[:idx].vstack(X[idx+1:])
            del doc_names[idx]
            del assigned_labels[idx]

    assigned_labels = np.array(assigned_labels, dtype=int)
    results = {}

    unique_labels = set(assigned_labels)
    if len(unique_labels) > 1:
        sil = compute_silhouette_score_custom(X, assigned_labels)
        sims, overall = compute_intra_label_similarity(X, assigned_labels)
        coh = compute_topic_coherence(cluster_map, papers_text)
        nmi_val = compute_nmi(X, assigned_labels, len(unique_labels), logger)
        results["silhouette_score"] = sil
        sims_str_keys = {str(k): float(v) for k, v in sims.items()}
        results["intra_label_similarity"] = sims_str_keys
        results["overall_avg_intra_label_similarity"] = overall
        results["topic_coherence"] = coh
        results["nmi_kmeans"] = nmi_val
    else:
        results["silhouette_score"] = None
        results["intra_label_similarity"] = {}
        results["overall_avg_intra_label_similarity"] = None
        results["topic_coherence"] = None
        results["nmi_kmeans"] = None

    with open(EVAL_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {EVAL_OUTPUT_JSON}")

if __name__ == "__main__":
    main()
