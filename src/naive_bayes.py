import numpy as np

# ÉTAPE 3 : LE MODÈLE NAIVE BAYES (L'intuition des comptes)
# Pourquoi ? C'est le modèle le plus simple pour le texte. 
# Il regarde : "Si le mot 'insulte' apparaît, quelle est la probabilité que ce soit toxique ?"

class MonNaiveBayes:
    def __init__(self, lissage=1.0):
        self.lissage = lissage # Sert à éviter les divisions par zéro
        self.probs_mots_toxiques = None
        self.probs_mots_sains = None
        self.prob_toxique = 0
        self.prob_sain = 0

    def fit(self, X, y):
        """
        Apprentissage : On compte les mots dans chaque catégorie.
        """
        n_echantillons, n_mots = X.shape
        
        # 1. Calculer la probabilité générale d'être toxique
        self.prob_toxique = np.mean(y)
        self.prob_sain = 1 - self.prob_toxique
        
        # 2. Séparer les données en deux groupes
        X_toxique = X[y == 1]
        X_sain = X[y == 0]
        
        # 3. Compter combien de fois chaque mot apparaît dans chaque groupe
        # On ajoute le 'lissage' pour ne pas avoir de probabilité de 0%
        comptes_toxiques = np.sum(X_toxique, axis=0) + self.lissage
        comptes_sains = np.sum(X_sain, axis=0) + self.lissage
        
        # 4. Convertir en probabilités
        self.probs_mots_toxiques = comptes_toxiques / np.sum(comptes_toxiques)
        self.probs_mots_sains = comptes_sains / np.sum(comptes_sains)

    def predict_proba(self, X):
        """
        Calcul du score de toxicité pour chaque phrase.
        """
        scores = []
        for i in range(X.shape[0]):
            # Pour chaque phrase, on multiplie les probabilités des mots présents
            # On utilise le logarithme (np.log) car multiplier des petits chiffres 
            # devient vite trop petit pour l'ordinateur.
            log_prob_tox = np.log(self.prob_toxique) + np.sum(X[i] * np.log(self.probs_mots_toxiques))
            log_prob_sain = np.log(self.prob_sain) + np.sum(X[i] * np.log(self.probs_mots_sains))
            
            # Plus le log_prob_tox est grand par rapport au log_prob_sain,
            # plus le commentaire est probablement toxique.
            
            # Pour simplifier, on transforme en score entre 0 et 1
            exp_tox = np.exp(log_prob_tox)
            exp_sain = np.exp(log_prob_sain)
            score = exp_tox / (exp_tox + exp_sain)
            scores.append(score)
            
        return np.array(scores)

    def predict(self, X, seuil=0.5):
        scores = self.predict_proba(X)
        return (scores >= seuil).astype(int)
