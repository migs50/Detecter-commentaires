import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def afficher_distribution_classes(df, colonne='est_toxique'):
    """EDA - Distribution des classes."""
    counts = df[colonne].value_counts()
    labels = ['Sain', 'Toxique']
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title("Distribution des Classes (EDA)")
    plt.ylabel("Nombre de commentaires")
    plt.savefig("eda_distribution.png")
    plt.close()
    print("📊 Graphique EDA sauvegardé")

def afficher_matrice_confusion(y_true, y_pred, nom_modele):
    """Matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)
    
    plt.title(f'Matrice de Confusion - {nom_modele}')
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.xticks([0, 1], ['Sain', 'Toxique'])
    plt.yticks([0, 1], ['Sain', 'Toxique'])
    plt.savefig(f"matrice_{nom_modele}.png")
    plt.close()

def afficher_histogramme(scores, titre="Scores de toxicité"):
    """Histogramme des scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(titre)
    plt.xlabel("Score (0=Sain, 1=Toxique)")
    plt.ylabel("Nombre de commentaires")
    plt.grid(axis='y', alpha=0.3)
    nom_fichier = titre.replace(" ", "_").lower() + ".png"
    plt.savefig(nom_fichier)
    plt.close()

def comparer_modeles(resultats):
    """Comparaison des modèles."""
    noms = list(resultats.keys())
    valeurs = list(resultats.values())
    
    plt.figure(figsize=(12, 6))
    colors = ['salmon', 'lightgreen', 'orange', 'cyan']
    plt.bar(noms, valeurs, color=colors[:len(noms)])
    plt.title("Comparaison des Modèles (F1-Score)")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("comparaison_modeles.png")
    plt.close()