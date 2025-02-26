import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def load_evaluation_results(llm_file: str, trad_file: str, logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Load evaluation results from both models"""
    try:
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_results = json.load(f)
        logger.info(f"Loaded LLM evaluation results from {llm_file}")
        
        with open(trad_file, 'r', encoding='utf-8') as f:
            trad_results = json.load(f)
        logger.info(f"Loaded traditional model evaluation results from {trad_file}")
        
        return llm_results, trad_results
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise

def compare_top_level_metrics(llm_results: Dict, trad_results: Dict, logger: logging.Logger) -> Dict:
    """Compare the top-level metrics between models"""
    metrics = {}
    
    # Overall density
    metrics['overall_density'] = {
        'llm': llm_results.get('overall_density', 0),
        'traditional': trad_results.get('overall_density', 0),
        'difference': llm_results.get('overall_density', 0) - trad_results.get('overall_density', 0),
        'better': 'LLM' if llm_results.get('overall_density', 0) > trad_results.get('overall_density', 0) else 'Traditional'
    }
    
    # Overall distinctiveness
    metrics['overall_distinctiveness'] = {
        'llm': llm_results.get('overall_distinctiveness', 0),
        'traditional': trad_results.get('overall_distinctiveness', 0),
        'difference': llm_results.get('overall_distinctiveness', 0) - trad_results.get('overall_distinctiveness', 0),
        'better': 'LLM' if llm_results.get('overall_distinctiveness', 0) > trad_results.get('overall_distinctiveness', 0) else 'Traditional'
    }
    
    # Coherence metrics
    for coh_type in ['c_v', 'c_uci', 'c_npmi']:
        llm_coh = llm_results.get('coherence_metrics', {}).get(coh_type)
        trad_coh = trad_results.get('coherence_metrics', {}).get(coh_type)
        
        if llm_coh is not None and trad_coh is not None:
            metrics[f'coherence_{coh_type}'] = {
                'llm': llm_coh,
                'traditional': trad_coh,
                'difference': llm_coh - trad_coh,
                'better': 'LLM' if llm_coh > trad_coh else 'Traditional'
            }
    
    # Overlap quality index
    llm_overlap = llm_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    trad_overlap = trad_results.get('overlap_metrics', {}).get('overlap_quality_index', 0)
    metrics['overlap_quality_index'] = {
        'llm': llm_overlap,
        'traditional': trad_overlap,
        'difference': llm_overlap - trad_overlap,
        'better': 'LLM' if llm_overlap > trad_overlap else 'Traditional'
    }
    
    # Label entropy (balance)
    llm_entropy = llm_results.get('balance_metrics', {}).get('label_entropy', 0)
    trad_entropy = trad_results.get('balance_metrics', {}).get('label_entropy', 0)
    metrics['label_entropy'] = {
        'llm': llm_entropy,
        'traditional': trad_entropy,
        'difference': llm_entropy - trad_entropy,
        'better': 'LLM' if llm_entropy > trad_entropy else 'Traditional'
    }
    
    # Average labels per document
    llm_avg_labels = llm_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    trad_avg_labels = trad_results.get('balance_metrics', {}).get('doc_label_stats', {}).get('mean', 0)
    metrics['avg_labels_per_doc'] = {
        'llm': llm_avg_labels,
        'traditional': trad_avg_labels,
        'difference': llm_avg_labels - trad_avg_labels,
        # This one is subjective - neither is necessarily "better"
        'note': 'More labels is not necessarily better, depends on use case'
    }
    
    # Count number of metrics where each model is better
    llm_wins = sum(1 for m in metrics.values() if m.get('better') == 'LLM')
    trad_wins = sum(1 for m in metrics.values() if m.get('better') == 'Traditional')
    
    # Log the comparison results
    logger.info("\n===== MODEL COMPARISON RESULTS =====")
    logger.info(f"LLM wins: {llm_wins} metrics")
    logger.info(f"Traditional wins: {trad_wins} metrics")
    
    for metric_name, result in metrics.items():
        if 'better' in result:
            logger.info(f"{metric_name}: LLM={result['llm']:.4f}, Traditional={result['traditional']:.4f}, Better: {result['better']}")
        else:
            logger.info(f"{metric_name}: LLM={result['llm']:.4f}, Traditional={result['traditional']:.4f}, Note: {result.get('note')}")
    
    return metrics

def compare_label_counts(llm_results: Dict, trad_results: Dict, logger: logging.Logger) -> Dict:
    """Compare the number of labels and documents in each model"""
    llm_label_count = llm_results.get('balance_metrics', {}).get('total_labels', 0)
    trad_label_count = trad_results.get('balance_metrics', {}).get('total_labels', 0)
    
    llm_doc_count = llm_results.get('balance_metrics', {}).get('total_docs', 0)
    trad_doc_count = trad_results.get('balance_metrics', {}).get('total_docs', 0)
    
    comparison = {
        'label_count': {
            'llm': llm_label_count,
            'traditional': trad_label_count,
            'difference': llm_label_count - trad_label_count
        },
        'doc_count': {
            'llm': llm_doc_count,
            'traditional': trad_doc_count,
            'difference': llm_doc_count - trad_doc_count
        }
    }
    
    logger.info("\n===== LABEL AND DOCUMENT COUNTS =====")
    logger.info(f"Number of labels: LLM={llm_label_count}, Traditional={trad_label_count}")
    logger.info(f"Number of documents: LLM={llm_doc_count}, Traditional={trad_doc_count}")
    
    return comparison

def create_visualizations(metrics: Dict, counts: Dict, output_dir: str, logger: logging.Logger):
    """Create visualizations comparing the models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart for main metrics
    plt.figure(figsize=(12, 8))
    metric_names = [m for m in metrics.keys() if m != 'avg_labels_per_doc']  # Exclude this one since it's subjective
    llm_values = [metrics[m]['llm'] for m in metric_names]
    trad_values = [metrics[m]['traditional'] for m in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, llm_values, width, label='LLM')
    rects2 = ax.bar(x + width/2, trad_values, width, label='Traditional')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Comparison of Key Metrics Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    chart_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(chart_path)
    logger.info(f"Saved metrics comparison chart to {chart_path}")
    plt.close()
    
    # 2. Radar chart for normalized metrics
    # Normalize metrics to scale of 0-1 for radar chart
    normalized_metrics = {}
    for metric_name in metric_names:
        max_val = max(metrics[metric_name]['llm'], metrics[metric_name]['traditional'])
        if max_val == 0:
            normalized_metrics[metric_name] = {'llm': 0, 'traditional': 0}
        else:
            normalized_metrics[metric_name] = {
                'llm': metrics[metric_name]['llm'] / max_val,
                'traditional': metrics[metric_name]['traditional'] / max_val
            }
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    llm_values = [normalized_metrics[m]['llm'] for m in metric_names]
    llm_values += llm_values[:1]  # Close the loop
    
    trad_values = [normalized_metrics[m]['traditional'] for m in metric_names]
    trad_values += trad_values[:1]  # Close the loop
    
    ax.plot(angles, llm_values, 'o-', linewidth=2, label='LLM')
    ax.fill(angles, llm_values, alpha=0.25)
    
    ax.plot(angles, trad_values, 'o-', linewidth=2, label='Traditional')
    ax.fill(angles, trad_values, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), [m.replace('_', ' ').title() for m in metric_names])
    ax.set_title('Normalized Metrics Comparison (Higher is Better)')
    ax.grid(True)
    ax.legend(loc='upper right')
    
    radar_path = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(radar_path)
    logger.info(f"Saved radar comparison chart to {radar_path}")
    plt.close()
    
    # 3. Pie chart of wins
    llm_wins = sum(1 for m in metrics.values() if m.get('better') == 'LLM')
    trad_wins = sum(1 for m in metrics.values() if m.get('better') == 'Traditional')
    
    plt.figure(figsize=(8, 8))
    plt.pie([llm_wins, trad_wins], labels=['LLM', 'Traditional'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    plt.axis('equal')
    plt.title('Percentage of Metrics Where Each Model Performs Better')
    
    pie_path = os.path.join(output_dir, 'wins_pie_chart.png')
    plt.savefig(pie_path)
    logger.info(f"Saved wins pie chart to {pie_path}")
    plt.close()

def analyze_label_quality(llm_results: Dict, trad_results: Dict, logger: logging.Logger) -> Dict:
    """Analyze the quality of individual labels"""
    
    # Compare label densities
    llm_densities = llm_results.get('label_densities', {})
    trad_densities = trad_results.get('label_densities', {})
    
    llm_avg_density = np.mean(list(llm_densities.values())) if llm_densities else 0
    trad_avg_density = np.mean(list(trad_densities.values())) if trad_densities else 0
    
    # Compare label distinctiveness
    llm_distinctiveness = llm_results.get('topic_distinctiveness', {})
    trad_distinctiveness = trad_results.get('topic_distinctiveness', {})
    
    llm_avg_distinctiveness = np.mean(list(llm_distinctiveness.values())) if llm_distinctiveness else 0
    trad_avg_distinctiveness = np.mean(list(trad_distinctiveness.values())) if trad_distinctiveness else 0
    
    # Analyze top words - how many unique top words across all labels?
    llm_top_words = llm_results.get('top_words_per_label', {})
    trad_top_words = trad_results.get('top_words_per_label', {})
    
    llm_all_top_words = set()
    for words in llm_top_words.values():
        llm_all_top_words.update(words[:5])  # Look at top 5 words per label
    
    trad_all_top_words = set()
    for words in trad_top_words.values():
        trad_all_top_words.update(words[:5])  # Look at top 5 words per label
    
    analysis = {
        'avg_label_density': {
            'llm': llm_avg_density,
            'traditional': trad_avg_density,
            'better': 'LLM' if llm_avg_density > trad_avg_density else 'Traditional'
        },
        'avg_label_distinctiveness': {
            'llm': llm_avg_distinctiveness,
            'traditional': trad_avg_distinctiveness,
            'better': 'LLM' if llm_avg_distinctiveness > trad_avg_distinctiveness else 'Traditional'
        },
        'unique_top_words_count': {
            'llm': len(llm_all_top_words),
            'traditional': len(trad_all_top_words),
            'better': 'LLM' if len(llm_all_top_words) > len(trad_all_top_words) else 'Traditional'
        }
    }
    
    logger.info("\n===== LABEL QUALITY ANALYSIS =====")
    logger.info(f"Average label density: LLM={llm_avg_density:.4f}, Traditional={trad_avg_density:.4f}")
    logger.info(f"Average label distinctiveness: LLM={llm_avg_distinctiveness:.4f}, Traditional={trad_avg_distinctiveness:.4f}")
    logger.info(f"Unique top words count: LLM={len(llm_all_top_words)}, Traditional={len(trad_all_top_words)}")
    
    return analysis

def generate_html_report(metrics: Dict, counts: Dict, label_quality: Dict, llm_results: Dict, trad_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate an HTML report summarizing the comparison"""
    
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
            .better { font-weight: bold; color: green; }
            .worse { color: #666; }
            .images { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0; }
            .image-container { text-align: center; }
            .summary { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Report: LLM vs. Traditional Approach</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
    """
    
    # Count wins
    llm_wins = sum(1 for m in metrics.values() if m.get('better') == 'LLM')
    trad_wins = sum(1 for m in metrics.values() if m.get('better') == 'Traditional')
    
    overall_winner = "LLM" if llm_wins > trad_wins else "Traditional" if trad_wins > llm_wins else "Tie"
    
    html_content += f"""
                <p>Based on {llm_wins + trad_wins} comparative metrics:</p>
                <ul>
                    <li>LLM approach wins on {llm_wins} metrics</li>
                    <li>Traditional approach wins on {trad_wins} metrics</li>
                    <li>Overall recommendation: <strong>{overall_winner}</strong></li>
                </ul>
            </div>
            
            <h2>Key Metrics Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>LLM</th>
                    <th>Traditional</th>
                    <th>Difference</th>
                    <th>Better Model</th>
                </tr>
    """
    
    # Add metrics to table
    for metric_name, values in metrics.items():
        if 'better' in values:
            llm_class = "better" if values['better'] == 'LLM' else "worse"
            trad_class = "better" if values['better'] == 'Traditional' else "worse"
            
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td class="{llm_class}">{values['llm']:.4f}</td>
                    <td class="{trad_class}">{values['traditional']:.4f}</td>
                    <td>{values['difference']:.4f}</td>
                    <td>{values['better']}</td>
                </tr>
            """
        else:
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td>{values['llm']:.4f}</td>
                    <td>{values['traditional']:.4f}</td>
                    <td>{values['difference']:.4f}</td>
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
                    <th>Difference</th>
                </tr>
    """
    
    # Add counts to table
    for count_name, values in counts.items():
        html_content += f"""
            <tr>
                <td>{count_name.replace('_', ' ').title()}</td>
                <td>{values['llm']}</td>
                <td>{values['traditional']}</td>
                <td>{values['difference']}</td>
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
                    <th>Better Model</th>
                </tr>
    """
    
    # Add label quality metrics
    for quality_name, values in label_quality.items():
        llm_class = "better" if values['better'] == 'LLM' else "worse"
        trad_class = "better" if values['better'] == 'Traditional' else "worse"
        
        html_content += f"""
            <tr>
                <td>{quality_name.replace('_', ' ').title()}</td>
                <td class="{llm_class}">{values['llm'] if isinstance(values['llm'], int) else values['llm']:.4f}</td>
                <td class="{trad_class}">{values['traditional'] if isinstance(values['traditional'], int) else values['traditional']:.4f}</td>
                <td>{values['better']}</td>
            </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <div class="images">
                <div class="image-container">
                    <img src="metrics_comparison.png" alt="Metrics Comparison" width="600">
                    <p>Comparison of Key Metrics</p>
                </div>
                <div class="image-container">
                    <img src="radar_comparison.png" alt="Radar Comparison" width="500">
                    <p>Normalized Metrics Comparison</p>
                </div>
                <div class="image-container">
                    <img src="wins_pie_chart.png" alt="Wins Distribution" width="400">
                    <p>Distribution of Better Performance</p>
                </div>
            </div>
            
            <h2>Detailed Model Information</h2>
            
            <h3>LLM Model Labels</h3>
            <ul>
    """
    
    # List LLM labels and their sizes
    llm_label_sizes = {label: len(docs) for label, docs in llm_results.get('top_words_per_label', {}).items()}
    sorted_llm_labels = sorted(llm_label_sizes.items(), key=lambda x: x[1], reverse=True)
    
    for label, size in sorted_llm_labels[:10]:  # Show top 10
        top_words = llm_results.get('top_words_per_label', {}).get(label, [])[:5]
        html_content += f"""
            <li><strong>{label}</strong> ({size} documents) - Top words: {', '.join(top_words)}</li>
        """
    
    html_content += """
            </ul>
            
            <h3>Traditional Model Labels</h3>
            <ul>
    """
    
    # List Traditional labels and their sizes
    trad_label_sizes = {label: len(docs) for label, docs in trad_results.get('top_words_per_label', {}).items()}
    sorted_trad_labels = sorted(trad_label_sizes.items(), key=lambda x: x[1], reverse=True)
    
    for label, size in sorted_trad_labels[:10]:  # Show top 10
        top_words = trad_results.get('top_words_per_label', {}).get(label, [])[:5]
        html_content += f"""
            <li><strong>{label}</strong> ({size} documents) - Top words: {', '.join(top_words)}</li>
        """
    
    html_content += """
            </ul>
            
            <h2>Conclusion</h2>
    """
    
    # Generate conclusion based on results
    if overall_winner == "LLM":
        html_content += """
            <p>The LLM-based approach demonstrates superior performance overall, particularly in metrics related to 
            semantic coherence and quality of clustering. This suggests that for this specific document set, 
            the language model's understanding of context and semantics provides more meaningful topic groupings.</p>
        """
    elif overall_winner == "Traditional":
        html_content += """
            <p>The traditional BERTopic approach demonstrates superior performance overall, particularly in metrics related to 
            cluster density and distinctiveness. This suggests that for this specific document set, 
            the statistical approach to topic modeling is more effective at creating well-separated topic clusters.</p>
        """
    else:
        html_content += """
            <p>Both approaches demonstrate comparable performance, with each excelling in different areas. 
            The choice between the two would depend on the specific requirements of the application, 
            such as whether semantic coherence or cluster separation is more important.</p>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    html_path = os.path.join(output_dir, 'model_comparison_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {html_path}")
    
    return html_path

def main():
    # Paths to evaluation results
    LLM_EVAL_PATH = r"DS1_Paper\llm_3_evaluation_results.json"
    TRADITIONAL_EVAL_PATH = r"DS1_Paper\trad-2-evaluation_results.json"
    OUTPUT_DIR = r"DS1_Paper/comparison_results"
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model comparison")
    
    # Load evaluation results
    llm_results, trad_results = load_evaluation_results(LLM_EVAL_PATH, TRADITIONAL_EVAL_PATH, logger)
    
    # Compare metrics
    metrics = compare_top_level_metrics(llm_results, trad_results, logger)
    
    # Compare label counts
    counts = compare_label_counts(llm_results, trad_results, logger)
    
    # Analyze label quality
    label_quality = analyze_label_quality(llm_results, trad_results, logger)
    
    # Create visualizations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_visualizations(metrics, counts, OUTPUT_DIR, logger)
    
    # Generate HTML report
    html_path = generate_html_report(metrics, counts, label_quality, llm_results, trad_results, OUTPUT_DIR, logger)
    
    logger.info(f"Comparison complete. HTML report available at: {html_path}")

if __name__ == "__main__":
    main()