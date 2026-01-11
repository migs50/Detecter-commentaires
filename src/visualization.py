import matplotlib.pyplot as plt

# ÉTAPE 5 : LA VISUALISATION (Les Graphes)
# Pourquoi ? Une image vaut mille mots. On veut voir la répartition des scores.

def afficher_histogramme(scores, titre="Répartition des scores de toxicité"):
    """
    Crée un histogramme simple.
    Les scores sont entre 0 (Sain) et 1 (Toxique).
    """
    plt.figure(figsize=(10, 6))
    
    # On crée l'histogramme avec 20 "bacs" (bins)
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    
    plt.title(titre)
    plt.xlabel("Score de toxicité (0 = Sain, 1 = Toxique)")
    plt.ylabel("Nombre de commentaires")
    
    # On affiche une grille pour mieux lire
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Sauvegarde de l'image
    nom_fichier = titre.replace(" ", "_").lower() + ".png"
    plt.savefig(nom_fichier)
    print(f"Graphique sauvegardé : {nom_fichier}")
    plt.close()

def afficher_distribution_classes(df, colonne='est_toxique'):
    """
    ÉTAPE : ANALYSE EXPLORATOIRE (EDA)
    Affiche si le dataset est équilibré (autant de toxiques que de sains).
    """
    counts = df[colonne].value_counts()
    labels = ['Sain (0)', 'Toxique (1)']
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title("Distribution des classes (Équilibre du dataset)")
    plt.ylabel("Nombre de commentaires")
    
    plt.savefig("distribution_classes.png")
    plt.close()
    print("Graphique EDA sauvegardé : distribution_classes.png")

def comparer_modeles(resultats):
    """
    Affiche une comparaison simple des précisions.
    resultats est un dictionnaire : {'Modèle': precision}
    """
    noms = list(resultats.keys())
    valeurs = list(resultats.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(noms, valeurs, color=['salmon', 'lightgreen', 'orange', 'cyan'])
    
    plt.title("Comparaison de la précision des modèles")
    plt.ylabel("Précision (Accuracy)")
    plt.ylim(0, 1) # L'échelle va de 0 à 100%
    
    plt.savefig("comparaison_modeles.png")
    plt.close()
