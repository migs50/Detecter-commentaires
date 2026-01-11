import numpy as np

# ÉTAPE 2 : VECTORISATION + NORMALISATION

class SimpleBagOfWords:
    """
    Transforme les textes en nombres (Bag of Words).
    Inclut une normalisation optionnelle (division par la somme).
    """
    def __init__(self, max_mots=1000, normaliser=True):
        self.max_mots = max_mots
        self.normaliser = normaliser
        self.vocabulaire = {}
        
    def fit(self, liste_textes):
        """Crée le vocabulaire des mots les plus fréquents."""
        frequences = {}
        for texte in liste_textes:
            for mot in texte.split():
                frequences[mot] = frequences.get(mot, 0) + 1
        
        # Garder les max_mots plus fréquents
        mots_tries = sorted(frequences.items(), key=lambda x: x[1], reverse=True)
        top_mots = [item[0] for item in mots_tries[:self.max_mots]]
        
        for i, mot in enumerate(top_mots):
            self.vocabulaire[mot] = i
            
        print(f"📚 Vocabulaire créé : {len(self.vocabulaire)} mots")

    def transform(self, liste_textes):
        """Transforme les textes en matrice de nombres."""
        n_phrases = len(liste_textes)
        n_mots = len(self.vocabulaire)
        matrice = np.zeros((n_phrases, n_mots))
        
        # Compter les occurrences
        for i, texte in enumerate(liste_textes):
            for mot in texte.split():
                if mot in self.vocabulaire:
                    index = self.vocabulaire[mot]
                    matrice[i, index] += 1
        
        # NORMALISATION (optionnelle)
        if self.normaliser:
            sommes = matrice.sum(axis=1, keepdims=True)
            sommes[sommes == 0] = 1  # Éviter division par zéro
            matrice = matrice / sommes
            print("✅ Normalisation appliquée")
            
        return matrice

    def fit_transform(self, liste_textes):
        """Combine fit() et transform()."""
        self.fit(liste_textes)
        return self.transform(liste_textes)