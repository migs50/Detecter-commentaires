import numpy as np

# ÉTAPE 2 : LA VECTORISATION (Le "Sac de Mots" ou Bag of Words)
# Pourquoi cette étape ? 
# Un algorithme de Machine Learning ne sait faire que des calculs sur des NOMBRES.
# Il ne comprend pas le sens des mots.
# On transforme donc chaque phrase en une liste de nombres qui compte la présence des mots.

class SimpleBagOfWords:
    def __init__(self, max_mots=1000):
        self.max_mots = max_mots
        self.vocabulaire = {} # Dictionnaire : { 'mot': index }
        
    def fit(self, liste_textes):
        """
        Apprend quels sont les mots les plus fréquents pour créer un dictionnaire.
        """
        frequences = {}
        for texte in liste_textes:
            for mot in texte.split():
                frequences[mot] = frequences.get(mot, 0) + 1
        
        # On trie par fréquence et on garde les 'max_mots' meilleurs
        mots_tries = sorted(frequences.items(), key=lambda x: x[1], reverse=True)
        top_mots = [item[0] for item in mots_tries[:self.max_mots]]
        
        # On crée le dictionnaire final
        for i, mot in enumerate(top_mots):
            self.vocabulaire[mot] = i
            
        print(f"Vocabulaire créé avec {len(self.vocabulaire)} mots.")

    def transform(self, liste_textes):
        """
        Transforme des phrases en un tableau de chiffres (X).
        """
        n_phrases = len(liste_textes)
        n_mots = len(self.vocabulaire)
        
        # On crée une matrice de zéros (Lignes = Phrases, Colonnes = Mots)
        matrice = np.zeros((n_phrases, n_mots))
        
        for i, texte in enumerate(liste_textes):
            for mot in texte.split():
                if mot in self.vocabulaire:
                    index = self.vocabulaire[mot]
                    matrice[i, index] += 1 # On compte l'apparition
                    
        return matrice

    def fit_transform(self, liste_textes):
        self.fit(liste_textes)
        return self.transform(liste_textes)
