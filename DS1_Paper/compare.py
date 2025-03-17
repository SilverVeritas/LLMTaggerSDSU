import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
import sys
from datetime import datetime

def setup_logging(log_dir="logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('model_comparison')
    logger.setLevel(logging.DEBUG)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'model_comparison_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_evaluation_results(llm_file: str, trad_file: str, no_tagging_file: str, summary_file: str, logger: logging.Logger) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load evaluation results from all four models"""
    try:
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_results = json.load(f)
        logger.info(f"Loaded LLM evaluation results from {llm_file}")
        
        with open(trad_file, 'r', encoding='utf-8') as f:
            trad_results = json.load(f)
        logger.info(f"Loaded traditional model evaluation results from {trad_file}")
        
        with open(no_tagging_file, 'r', encoding='utf-8') as f:
            no_tagging_results = json.load(f)
        logger.info(f"Loaded no-tagging model evaluation results from {no_tagging_file}")
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_results = json.load(f)
        logger.info(f"Loaded summary approach evaluation results from {summary_file}")
        
        return llm_results, trad_results, no_tagging_results, summary_results
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise

def compare_top_level_metrics(llm_results: Dict, trad_results: Dict, no_tagging_results: Dict, summary_results: Dict, logger: logging.Logger) -> Dict:
    """Compare the top-level metrics between all four models"""
    metrics = {}
    
    # Overall density
    metrics['overall_density'] = {
        'llm': llm_results.get('overall_density', 0),
        'traditional': trad_results.get('overall_density', 0),
        'no_tagging': no_tagging_results.get('overall_density', 0),
        'summary': summary_results.get('overall_density', 0)
    }
    
    # Overall distinctiveness
    metrics['overall_distinctiveness'] = {
        'llm': llm_results.get('overall_distinctiveness', 0),
        'traditional': trad_results.get('overall_distinctiveness', 0),
        'no_tagging': no_tagging_results.get('overall_distinctiveness', 0),
        'summary': summary_results.get('overall_distinctiveness', 0)
    }
    
    # Coherence metrics
    for coh_type in ['c_v', 'c_uci', 'c_npmi']:
        llm_coh = llm_results.get('coherence_metrics', {}).get(coh_type)
        trad_coh = trad_results.get('coherence_metrics', {}).get(coh_type)
        no_tagging_coh = no_tagging_results.get('coherence_metrics', {}).get(coh_type)
        summary_coh = summary_results.get('coherence_metrics', {}).get(coh_type)
        
        if llm_coh is not None and trad_coh is not None and no_tagging_coh is not None and summary_coh is not None:
            metrics[f'coherence_{coh_type}'] = {
                'llm': llm_coh,
                'traditional': trad_coh,
                'no_tagging': no_tagging_coh,
                'summary': summary_coh
            }
    
    # Overlap quality index
    llm_overlap = llm_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    trad_overlap = trad_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    no_tagging_overlap = no_tagging_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    summary_overlap = summary_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    metrics['overlap_quality_index'] = {
        'llm': llm_overlap,
        'traditional': trad_overlap,
        'no_tagging': no_tagging_overlap,
        'summary': summary_overlap
    }
    
    # Label entropy (balance)
    llm_entropy = llm_results.get('balance_metrics', {}).get('label_entropy', 0)
    trad_entropy = trad_results.get('balance_metrics', {}).get('label_entropy', 0)
    no_tagging_entropy = no_tagging_results.get('balance_metrics', {}).get('label_entropy', 0)
    summary_entropy = summary_results.get('balance_metrics', {}).get('label_entropy', 0)
    metrics['label_entropy'] = {
        'llm': llm_entropy,
        'traditional': trad_entropy,
        'no_tagging': no_tagging_entropy,
        'summary': summary_entropy
    }
    
    # Average labels per document
    llm_avg_labels = llm_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    trad_avg_labels = trad_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    no_tagging_avg_labels = no_tagging_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    summary_avg_labels = summary_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    metrics['avg_labels_per_doc'] = {
        'llm': llm_avg_labels,
        'traditional': trad_avg_labels,
        'no_tagging': no_tagging_avg_labels,
        'summary': summary_avg_labels,
        'note': 'More labels is not necessarily better, depends on use case'
    }
    
    # Find which model performs best for each metric
    for metric_name, values in metrics.items():
        if metric_name != 'avg_labels_per_doc':  # Skip this one as it's subjective
            metric_values = {k: v for k, v in values.items() if k in ['llm', 'traditional', 'no_tagging', 'summary']}
            best_model = max(metric_values, key=metric_values.get)
            values['best'] = best_model
    
    # Count number of metrics where each model is best
    llm_wins = sum(1 for m in metrics.values() if m.get('best') == 'llm')
    trad_wins = sum(1 for m in metrics.values() if m.get('best') == 'traditional')
    no_tagging_wins = sum(1 for m in metrics.values() if m.get('best') == 'no_tagging')
    summary_wins = sum(1 for m in metrics.values() if m.get('best') == 'summary')
    
    # Log the comparison results
    logger.info("\n===== MODEL COMPARISON RESULTS =====")
    logger.info(f"LLM wins: {llm_wins} metrics")
    logger.info(f"Traditional wins: {trad_wins} metrics")
    logger.info(f"No-Tagging wins: {no_tagging_wins} metrics")
    logger.info(f"Summary wins: {summary_wins} metrics")
    
    for metric_name, result in metrics.items():
        if 'best' in result:
            logger.info(f"{metric_name}: LLM={result['llm']:.4f}, Traditional={result['traditional']:.4f}, " + 
                       f"No-Tagging={result['no_tagging']:.4f}, Summary={result['summary']:.4f}, Best: {result['best']}")
        else:
            logger.info(f"{metric_name}: LLM={result['llm']:.4f}, Traditional={result['traditional']:.4f}, " + 
                       f"No-Tagging={result['no_tagging']:.4f}, Summary={result['summary']:.4f}, Note: {result.get('note')}")
    
    return metrics

def compare_label_counts(llm_results: Dict, trad_results: Dict, no_tagging_results: Dict, summary_results: Dict, logger: logging.Logger) -> Dict:
    """Compare the number of labels and documents in each model"""
    llm_label_count = llm_results.get('balance_metrics', {}).get('total_labels', 0)
    trad_label_count = trad_results.get('balance_metrics', {}).get('total_labels', 0)
    no_tagging_label_count = no_tagging_results.get('balance_metrics', {}).get('total_labels', 0)
    summary_label_count = summary_results.get('balance_metrics', {}).get('total_labels', 0)
    
    llm_doc_count = llm_results.get('balance_metrics', {}).get('total_docs', 0)
    trad_doc_count = trad_results.get('balance_metrics', {}).get('total_docs', 0)
    no_tagging_doc_count = no_tagging_results.get('balance_metrics', {}).get('total_docs', 0)
    summary_doc_count = summary_results.get('balance_metrics', {}).get('total_docs', 0)
    
    comparison = {
        'label_count': {
            'llm': llm_label_count,
            'traditional': trad_label_count,
            'no_tagging': no_tagging_label_count,
            'summary': summary_label_count
        },
        'doc_count': {
            'llm': llm_doc_count,
            'traditional': trad_doc_count,
            'no_tagging': no_tagging_doc_count,
            'summary': summary_doc_count
        }
    }
    
    logger.info("\n===== LABEL AND DOCUMENT COUNTS =====")
    logger.info(f"Number of labels: LLM={llm_label_count}, Traditional={trad_label_count}, " + 
               f"No-Tagging={no_tagging_label_count}, Summary={summary_label_count}")
    logger.info(f"Number of documents: LLM={llm_doc_count}, Traditional={trad_doc_count}, " + 
               f"No-Tagging={no_tagging_doc_count}, Summary={summary_doc_count}")
    
    return comparison

def analyze_label_quality(llm_results: Dict, trad_results: Dict, no_tagging_results: Dict, summary_results: Dict, logger: logging.Logger) -> Dict:
    """Analyze the quality of individual labels across all four models"""
    
    # Compare label densities
    llm_densities = llm_results.get('label_densities', {})
    trad_densities = trad_results.get('label_densities', {})
    no_tagging_densities = no_tagging_results.get('label_densities', {})
    summary_densities = summary_results.get('label_densities', {})
    
    llm_avg_density = np.mean(list(llm_densities.values())) if llm_densities else 0
    trad_avg_density = np.mean(list(trad_densities.values())) if trad_densities else 0
    no_tagging_avg_density = np.mean(list(no_tagging_densities.values())) if no_tagging_densities else 0
    summary_avg_density = np.mean(list(summary_densities.values())) if summary_densities else 0
    
    # Compare label distinctiveness
    llm_distinctiveness = llm_results.get('topic_distinctiveness', {})
    trad_distinctiveness = trad_results.get('topic_distinctiveness', {})
    no_tagging_distinctiveness = no_tagging_results.get('topic_distinctiveness', {})
    summary_distinctiveness = summary_results.get('topic_distinctiveness', {})
    
    llm_avg_distinctiveness = np.mean(list(llm_distinctiveness.values())) if llm_distinctiveness else 0
    trad_avg_distinctiveness = np.mean(list(trad_distinctiveness.values())) if trad_distinctiveness else 0
    no_tagging_avg_distinctiveness = np.mean(list(no_tagging_distinctiveness.values())) if no_tagging_distinctiveness else 0
    summary_avg_distinctiveness = np.mean(list(summary_distinctiveness.values())) if summary_distinctiveness else 0
    
    # Find best model for each metric
    density_best = max(['llm', 'traditional', 'no_tagging', 'summary'], 
                     key=lambda m: {'llm': llm_avg_density, 'traditional': trad_avg_density, 
                                  'no_tagging': no_tagging_avg_density, 'summary': summary_avg_density}[m])
    
    distinctiveness_best = max(['llm', 'traditional', 'no_tagging', 'summary'], 
                             key=lambda m: {'llm': llm_avg_distinctiveness, 'traditional': trad_avg_distinctiveness, 
                                          'no_tagging': no_tagging_avg_distinctiveness, 'summary': summary_avg_distinctiveness}[m])
    
    analysis = {
        'avg_label_density': {
            'llm': llm_avg_density,
            'traditional': trad_avg_density,
            'no_tagging': no_tagging_avg_density,
            'summary': summary_avg_density,
            'best': density_best
        },
        'avg_label_distinctiveness': {
            'llm': llm_avg_distinctiveness,
            'traditional': trad_avg_distinctiveness,
            'no_tagging': no_tagging_avg_distinctiveness,
            'summary': summary_avg_distinctiveness,
            'best': distinctiveness_best
        }
    }
    
    logger.info("\n===== LABEL QUALITY ANALYSIS =====")
    logger.info(f"Average label density: LLM={llm_avg_density:.4f}, Traditional={trad_avg_density:.4f}, " + 
               f"No-Tagging={no_tagging_avg_density:.4f}, Summary={summary_avg_density:.4f}, Best: {density_best}")
    logger.info(f"Average label distinctiveness: LLM={llm_avg_distinctiveness:.4f}, Traditional={trad_avg_distinctiveness:.4f}, " + 
               f"No-Tagging={no_tagging_avg_distinctiveness:.4f}, Summary={summary_avg_distinctiveness:.4f}, Best: {distinctiveness_best}")
    
    return analysis

def generate_html_report(metrics: Dict, counts: Dict, label_quality: Dict, output_dir: str, logger: logging.Logger):
    """Generate an HTML report summarizing the comparison without images"""
    
    # Create the HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .best { font-weight: bold; color: green; }
            .worse { color: #666; }
            .summary { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Report: All Approaches</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
    """
    
    # Count wins
    llm_wins = sum(1 for m in metrics.values() if m.get('best') == 'llm')
    trad_wins = sum(1 for m in metrics.values() if m.get('best') == 'traditional')
    no_tagging_wins = sum(1 for m in metrics.values() if m.get('best') == 'no_tagging')
    summary_wins = sum(1 for m in metrics.values() if m.get('best') == 'summary')
    
    # Determine overall winner
    model_wins = {
        'LLM': llm_wins,
        'Traditional': trad_wins,
        'No-Tagging': no_tagging_wins,
        'Summarization': summary_wins
    }
    overall_winner = max(model_wins, key=model_wins.get)
    if list(model_wins.values()).count(model_wins[overall_winner]) > 1:
        overall_winner = "Tie"
    
    html_content += f"""
                <p>Based on {llm_wins + trad_wins + no_tagging_wins + summary_wins} comparative metrics:</p>
                <ul>
                    <li>LLM approach (with keyword extraction) wins on {llm_wins} metrics</li>
                    <li>Traditional approach wins on {trad_wins} metrics</li>
                    <li>No-Tagging approach (direct document-to-label) wins on {no_tagging_wins} metrics</li>
                    <li>Summarization approach (summarize → extract → label) wins on {summary_wins} metrics</li>
                    <li>Overall recommendation: <strong>{overall_winner}</strong></li>
                </ul>
            </div>
            
            <h2>Key Metrics Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>LLM</th>
                    <th>Traditional</th>
                    <th>No-Tagging</th>
                    <th>Summarization</th>
                    <th>Best Model</th>
                </tr>
    """
    
    # Add metrics to table
    for metric_name, values in metrics.items():
        if 'best' in values:
            best_model = values['best']
            llm_class = "best" if best_model == 'llm' else "worse"
            trad_class = "best" if best_model == 'traditional' else "worse"
            no_tagging_class = "best" if best_model == 'no_tagging' else "worse"
            summary_class = "best" if best_model == 'summary' else "worse"
            
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td class="{llm_class}">{values['llm']:.4f}</td>
                    <td class="{trad_class}">{values['traditional']:.4f}</td>
                    <td class="{no_tagging_class}">{values['no_tagging']:.4f}</td>
                    <td class="{summary_class}">{values['summary']:.4f}</td>
                    <td>{best_model.replace('_', ' ').title()}</td>
                </tr>
            """
        else:
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td>{values['llm']:.4f}</td>
                    <td>{values['traditional']:.4f}</td>
                    <td>{values['no_tagging']:.4f}</td>
                    <td>{values['summary']:.4f}</td>
                    <td>{values.get('note', 'N/A')}</td>
                </tr>
            """
    
    html_content += """
            </table>
            
            <h2>Label and Document Counts</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>LLM</th>
                    <th>Traditional</th>
                    <th>No-Tagging</th>
                    <th>Summarization</th>
                </tr>
    """
    
    # Add counts to table
    for count_name, values in counts.items():
        html_content += f"""
            <tr>
                <td>{count_name.replace('_', ' ').title()}</td>
                <td>{values['llm']}</td>
                <td>{values['traditional']}</td>
                <td>{values['no_tagging']}</td>
                <td>{values['summary']}</td>
            </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Label Quality Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>LLM</th>
                    <th>Traditional</th>
                    <th>No-Tagging</th>
                    <th>Summarization</th>
                    <th>Best Model</th>
                </tr>
    """
    
    # Add label quality metrics
    for quality_name, values in label_quality.items():
        best_model = values['best']
        llm_class = "best" if best_model == 'llm' else "worse"
        trad_class = "best" if best_model == 'traditional' else "worse"
        no_tagging_class = "best" if best_model == 'no_tagging' else "worse"
        summary_class = "best" if best_model == 'summary' else "worse"
        
        html_content += f"""
            <tr>
                <td>{quality_name.replace('_', ' ').title()}</td>
                <td class="{llm_class}">{values['llm'] if isinstance(values['llm'], int) else values['llm']:.4f}</td>
                <td class="{trad_class}">{values['traditional'] if isinstance(values['traditional'], int) else values['traditional']:.4f}</td>
                <td class="{no_tagging_class}">{values['no_tagging'] if isinstance(values['no_tagging'], int) else values['no_tagging']:.4f}</td>
                <td class="{summary_class}">{values['summary'] if isinstance(values['summary'], int) else values['summary']:.4f}</td>
                <td>{best_model.title()}</td>
            </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Approach Descriptions</h2>
            <table>
                <tr>
                    <th>Approach</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>LLM</td>
                    <td>Extract keywords from full documents using LLM, then generate labels from those keywords.</td>
                </tr>
                <tr>
                    <td>Traditional</td>
                    <td>Use BERTopic or similar statistical approach to generate topics from document embeddings.</td>
                </tr>
                <tr>
                    <td>No-Tagging</td>
                    <td>Send documents directly to LLM to generate labels without intermediate keyword extraction.</td>
                </tr>
                <tr>
                    <td>Summarization</td>
                    <td>First summarize documents using LLM, then extract keywords from summaries, then generate labels.</td>
                </tr>
            </table>
            
            <h2>Conclusion</h2>
    """
    
    # Generate conclusion based on results
    if overall_winner == "LLM":
        html_content += """
            <p>The LLM-based approach with keyword extraction demonstrates superior performance overall, particularly in metrics related to 
            semantic coherence and quality of clustering. This suggests that for this specific document set, 
            the two-step process of extracting keywords and then generating labels provides the most effective results.</p>
        """
    elif overall_winner == "Traditional":
        html_content += """
            <p>The traditional BERTopic approach demonstrates superior performance overall, particularly in metrics related to 
            cluster density and distinctiveness. This suggests that for this specific document set, 
            the statistical approach to topic modeling is more effective at creating well-separated topic clusters.</p>
        """
    elif overall_winner == "No-Tagging":
        html_content += """
            <p>The No-Tagging approach (direct document-to-label LLM) demonstrates superior performance overall, suggesting that 
            the intermediate keyword extraction step might not be necessary. The language model appears capable of understanding 
            the document content directly and generating more coherent and distinct topic labels.</p>
        """
    elif overall_winner == "Summarization":
        html_content += """
            <p>The Summarization approach (summarize → extract → label) demonstrates superior performance overall. This suggests that 
            condensing documents first removes noise and helps focus on the most relevant content, leading to better keyword extraction 
            and ultimately more effective labels. The three-step pipeline appears to provide the best balance of coherence, distinctiveness, 
            and quality for this document set.</p>
        """
    else:
        html_content += """
            <p>The four approaches demonstrate comparable performance, with each excelling in different areas. 
            The choice between them would depend on the specific requirements of the application, 
            such as whether semantic coherence, cluster separation, processing efficiency, or some combination is most important.</p>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    html_path = os.path.join(output_dir, 'DS1_Paper_model_comparison_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {html_path}")
    
    return html_path

def main():
    # Paths to evaluation results
    LLM_EVAL_PATH = r"DS1_Paper\llm_3_evaluation_results.json"
    TRADITIONAL_EVAL_PATH = r"DS1_Paper\trad-2-evaluation_results.json"
    NO_TAGGING_EVAL_PATH = r"DS1_Paper\no-tagging_evaluation_results.json"
    SUMMARY_EVAL_PATH = r"DS1_Paper\summary_evaluation_results.json"
    OUTPUT_DIR = r"DS1_Paper\comparison_results"
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model comparison for all four approaches")
    
    # Load evaluation results
    llm_results, trad_results, no_tagging_results, summary_results = load_evaluation_results(
        LLM_EVAL_PATH, TRADITIONAL_EVAL_PATH, NO_TAGGING_EVAL_PATH, SUMMARY_EVAL_PATH, logger
    )
    
    # Compare metrics
    metrics = compare_top_level_metrics(llm_results, trad_results, no_tagging_results, summary_results, logger)
    
    # Compare label counts
    counts = compare_label_counts(llm_results, trad_results, no_tagging_results, summary_results, logger)
    
    # Analyze label quality
    label_quality = analyze_label_quality(llm_results, trad_results, no_tagging_results, summary_results, logger)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate HTML report
    html_path = generate_html_report(metrics, counts, label_quality, OUTPUT_DIR, logger)
    
    logger.info(f"Comparison complete. HTML report available at: {html_path}")

if __name__ == "__main__":
    main()