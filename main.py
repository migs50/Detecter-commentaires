import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# Local imports
from src.preprocessing import preprocess_dataframe
from src.vectorization import get_tfidf_features
from src.naive_bayes import MultinomialNaiveBayes
from src.logistic_regression import LogisticRegression
from src.visualization import plot_class_distribution, plot_toxicity_scores, plot_metrics_comparison
from sklearn_models import train_sklearn_models

def main():
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv('data/comments.csv')
    
    # Use a subset for faster development/execution
    # 20,000 rows is a good balance
    df = df.sample(n=20000, random_state=42)
    
    # 2. Preprocessing
    print("Preprocessing data...")
    df = preprocess_dataframe(df)
    
    # 3. EDA - Class Distribution
    print("Visualizing class distribution...")
    plot_class_distribution(df)
    
    # 4. Split and Address Imbalance
    X = df['cleaned_text']
    y = df['is_toxic']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Vectorization
    print("Vectorizing text data...")
    X_train_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(X_train_raw, X_test_raw)
    
    # Handle Imbalance on training set
    print("Handling class imbalance with oversampling...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_tfidf, y_train)
    
    all_metrics = {}

    # 6. Train Scikit-Learn Models (Baseline)
    print("Training Scikit-Learn models...")
    sklearn_results = train_sklearn_models(X_train_res, y_train_res, X_test_tfidf, y_test)
    all_metrics.update(sklearn_results)

    # 7. Train Naive Bayes from Scratch
    print("Training Naive Bayes from scratch...")
    nb_scratch = MultinomialNaiveBayes()
    nb_scratch.fit(X_train_res, y_train_res)
    y_pred_nb = nb_scratch.predict(X_test_tfidf)
    y_prob_nb = nb_scratch.predict_proba(X_test_tfidf)[:, 1]
    
    all_metrics['NB_Scratch'] = {
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Precision': precision_score(y_test, y_pred_nb),
        'Recall': recall_score(y_test, y_pred_nb),
        'F1': f1_score(y_test, y_pred_nb),
        'probs': y_prob_nb
    }

    # 8. Train Logistic Regression from Scratch
    print("Training Logistic Regression from scratch (Gradient Descent)...")
    # For speed, we use fewer iterations and a larger learning rate or a smaller sample if needed
    lr_scratch = LogisticRegression(lr=0.1, n_iters=200) 
    lr_scratch.fit(X_train_res, y_train_res)
    y_pred_lr = lr_scratch.predict(X_test_tfidf)
    y_prob_lr = lr_scratch.predict_proba(X_test_tfidf)
    
    all_metrics['LR_Scratch'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'F1': f1_score(y_test, y_pred_lr),
        'probs': y_prob_lr
    }

    # 9. Final Evaluation & Visualization
    print("Generating final visualizations...")
    metrics_to_plot = {k: {m: v for m, v in s.items() if m != 'probs'} for k, s in all_metrics.items()}
    plot_metrics_comparison(metrics_to_plot)
    
    # Plot toxicity scores for the best performing scratch model (usually NB for text)
    plot_toxicity_scores(all_metrics['NB_Scratch']['probs'], title='Toxicity Scores - Naive Bayes Scratch')

    print("Pipeline completed sucessfully!")
    print("\nResults comparison:")
    for model, scores in metrics_to_plot.items():
        print(f"\n{model}:")
        for metric, val in scores.items():
            print(f"  {metric}: {val:.4f}")

if __name__ == "__main__":
    main()
