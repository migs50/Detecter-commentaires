import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(train_texts, test_texts, max_features=5000):
    """
    Convert text data into TF-IDF features.
    We'll use Scikit-Learn's TfidfVectorizer for efficiency, 
    but the focus of the task is the models from scratch.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    return X_train, X_test, vectorizer

class ScratchTFIDF:
    """
    Simple implementation of TF-IDF from scratch for educational purposes.
    (Optional/Bonus for comparison)
    """
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}

    def fit_transform(self, texts):
        # Implementation of fit and transform from scratch
        # This is a bit complex for a quick script but can be done if needed.
        pass
