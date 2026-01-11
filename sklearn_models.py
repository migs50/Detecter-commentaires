from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_sklearn_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Scikit-Learn models.
    """
    results = {}

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_prob_nb = nb.predict_proba(X_test)[:, 1]
    
    results['NB_Sklearn'] = {
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Precision': precision_score(y_test, y_pred_nb),
        'Recall': recall_score(y_test, y_pred_nb),
        'F1': f1_score(y_test, y_pred_nb),
        'probs': y_prob_nb
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results['LR_Sklearn'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'F1': f1_score(y_test, y_pred_lr),
        'probs': y_prob_lr
    }

    return results
