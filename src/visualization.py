import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_class_distribution(df, label_column='is_toxic'):
    """
    Plot the distribution of classes (Toxic vs Non-Toxic).
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=label_column, data=df, palette='viridis')
    plt.title('Distribution of Toxicity')
    plt.xlabel('Is Toxic (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()

def plot_toxicity_scores(y_probs, title='Histogram of Toxicity Scores'):
    """
    Plot a histogram of the predicted toxicity probabilities.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_probs, bins=50, kde=True, color='red')
    plt.title(title)
    plt.xlabel('Toxicity Score')
    plt.ylabel('Frequency')
    plt.savefig('toxicity_scores_hist.png')
    plt.close()

def plot_metrics_comparison(metrics_dict):
    """
    metrics_dict: { 'Model Name': {'Accuracy': 0.9, 'F1': 0.8, ...}, ... }
    """
    df_metrics = []
    for model, scores in metrics_dict.items():
        for metric, val in scores.items():
            df_metrics.append({'Model': model, 'Metric': metric, 'Score': val})
    
    df_metrics = pd.DataFrame(df_metrics)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_metrics, palette='magma')
    plt.title('Comparison of Model Metrics')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()
