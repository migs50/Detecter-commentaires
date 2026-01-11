import numpy as np

# ÉTAPE 5 : LA RÉGRESSION LOGISTIQUE
# Pourquoi ? Ce modèle apprend des "poids". Un mot très toxique aura un gros poids.

class MaRegressionLogistique:
    def __init__(self, taux_apprentissage=0.01, iterations=100):
        self.lr = taux_apprentissage
        self.it = iterations
        self.poids = None
        self.biais = 0

    def _sigmoid(self, z):
        """
        La fonction magique qui transforme n'importe quel chiffre en un score entre 0 et 1.
        Formule : sigmoid(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Apprentissage : On ajuste les poids petit à petit (Descente de Gradient).
        """
        n_echantillons, n_mots = X.shape
        self.poids = np.zeros(n_mots)  # Au début, on ne connaît rien (poids = 0)
        self.biais = 0

        print(f"🔄 Entraînement Régression Logistique ({self.it} itérations)...")

        for i in range(self.it):
            # 1. Prédiction actuelle : Score = (X * poids) + biais
            modele_lineaire = np.dot(X, self.poids) + self.biais
            predictions = self._sigmoid(modele_lineaire)

            # 2. Calculer l'erreur (Comment on doit changer les poids)
            erreur = predictions - y
            gradient_poids = (1 / n_echantillons) * np.dot(X.T, erreur)
            gradient_biais = (1 / n_echantillons) * np.sum(erreur)

            # 3. Mettre à jour les poids (On tourne les boutons)
            self.poids -= self.lr * gradient_poids
            self.biais -= self.lr * gradient_biais
            
            if i % 20 == 0:
                print(f"  Itération {i}...")

    def predict_proba(self, X):
        """
        Calcule le score de toxicité (entre 0 et 1) pour chaque commentaire.
        """
        modele_lineaire = np.dot(X, self.poids) + self.biais
        return self._sigmoid(modele_lineaire)

    def predict(self, X, seuil=0.5):
        """
        Prédit la classe (0 ou 1) en fonction d'un seuil.
        Si le score >= 0.5, on dit que c'est toxique (1), sinon sain (0).
        """
        return (self.predict_proba(X) >= seuil).astype(int)