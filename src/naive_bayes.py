import numpy as np

class MonNaiveBayes:
    def __init__(self, lissage=1.0):
        self.lissage = lissage
        self.probs_mots_toxiques = None
        self.probs_mots_sains = None
        self.prob_toxique = 0
        self.prob_sain = 0

    def fit(self, X, y):
        n_echantillons, n_mots = X.shape
        
        self.prob_toxique = np.mean(y)
        self.prob_sain = 1 - self.prob_toxique
        
        X_toxique = X[y == 1]
        X_sain = X[y == 0]
        
        comptes_toxiques = np.sum(X_toxique, axis=0) + self.lissage
        comptes_sains = np.sum(X_sain, axis=0) + self.lissage
        
        self.probs_mots_toxiques = comptes_toxiques / np.sum(comptes_toxiques)
        self.probs_mots_sains = comptes_sains / np.sum(comptes_sains)

    def predict_proba(self, X):
        scores = []
        for i in range(X.shape[0]):
            log_prob_tox = np.log(self.prob_toxique) + np.sum(X[i] * np.log(self.probs_mots_toxiques))
            log_prob_sain = np.log(self.prob_sain) + np.sum(X[i] * np.log(self.probs_mots_sains))
            
            exp_tox = np.exp(log_prob_tox)
            exp_sain = np.exp(log_prob_sain)
            score = exp_tox / (exp_tox + exp_sain)
            scores.append(score)
            
        return np.array(scores)

    def predict(self, X, seuil=0.5):
        return (self.predict_proba(X) >= seuil).astype(int)