import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplacian smoothing
        self.class_prior = None
        self.feature_log_prob = None
        self.classes = None

    def fit(self, X, y):
        """
        X: array-like or sparse matrix of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        y = np.array(y) # Ensure y is a numpy array
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize priors and feature probabilities
        self.class_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            # Samples belonging to class c
            X_c = X[y == c]
            
            # Prior P(c) = count(c) / count(total)
            self.class_prior[i] = X_c.shape[0] / n_samples
            
            # Sum of features for class c + smoothing
            feature_count = np.array(X_c.sum(axis=0)).flatten() + self.alpha
            
            # Total count of features in class c + smoothing * n_features
            total_count = feature_count.sum()
            
            # Log likelihood: log(P(x|c))
            self.feature_log_prob[i] = np.log(feature_count / total_count)

        # Log prior: log(P(c))
        self.class_log_prior = np.log(self.class_prior)

    def _joint_log_likelihood(self, X):
        """
        Calculate log(P(c)) + log(P(x|c))
        """
        # X * log_prob.T results in (n_samples, n_classes)
        return X @ self.feature_log_prob.T + self.class_log_prior

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        """
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        """
        jll = self._joint_log_likelihood(X)
        # Softmax of log likelihoods to get probabilities
        exp_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))
        return exp_jll / exp_jll.sum(axis=1, keepdims=True)
