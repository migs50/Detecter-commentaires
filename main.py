import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Imports des fichiers du projet
from src.preprocessing import preparer_donnees, equilibrer_donnees
from src.vectorization import SimpleBagOfWords
from src.naive_bayes import MonNaiveBayes
from src.logistic_regression import MaRegressionLogistique
from src.visualization import afficher_histogramme, comparer_modeles, afficher_distribution_classes
from sklearn_models import entrainer_modeles_pro

def main():
    print("=== PROJET ML : DÉTECTION DE TOXICITÉ (Version Académique) ===\n")

    # 1. CHARGEMENT DES DONNÉES
    print("1. Chargement des données...")
    try:
        df = pd.read_csv('data/archive/train.csv')
    except FileNotFoundError:
        print("Erreur : Le fichier 'data/archive/train.csv' est introuvable.")
        return

    # On prend un échantillon plus large pour avoir assez de toxiques après équilibrage
    df = df.sample(n=10000, random_state=42) 

    # 2. PRÉTRAITEMENT & MISSING VALUES
    print("2. Prétraitement et gestion des valeurs manquantes...")
    df = preparer_donnees(df)

    # 3. ANALYSE EXPLORATOIRE (EDA)
    print("3. Analyse exploratoire (EDA)...")
    afficher_distribution_classes(df)

    # 4. GESTION DU DÉSÉQUILIBRE DES CLASSES
    print("4. Gestion du déséquilibre (Undersampling)...")
    df = equilibrer_donnees(df)
    # On vérifie la nouvelle distribution
    print(f"Nouveau nombre de lignes après équilibrage : {len(df)}")

    # 5. SÉPARATION (Entraînement / Test)
    X_train_brut, X_test_brut, y_train, y_test = train_test_split(
        df['texte_nettoye'], df['est_toxique'], test_size=0.2, random_state=42
    )

    # 6. VECTORISATION (Bag of Words)
    print("\n6. Vectorisation des textes...")
    bow = SimpleBagOfWords(max_mots=1000)
    X_train = bow.fit_transform(X_train_brut)
    X_test = bow.transform(X_test_brut)

    # 7. ENTRAÎNEMENT DES MODÈLES (Fait Maison)
    resultats_finaux = {}

    # --- Naive Bayes ---
    print("\n7. Entraînement de Naive Bayes (Maison)...")
    model_nb = MonNaiveBayes()
    model_nb.fit(X_train, y_train.values)
    scores_nb = model_nb.predict_proba(X_test)
    pred_nb = model_nb.predict(X_test)
    
    acc_nb = accuracy_score(y_test, pred_nb)
    f1_nb = f1_score(y_test, pred_nb)
    resultats_finaux['NB (Maison) Acc'] = acc_nb
    resultats_finaux['NB (Maison) F1'] = f1_nb
    print(f"  Accuracy : {acc_nb:.2%} | F1-Score : {f1_nb:.2%}")

    # --- Régression Logistique ---
    print("\n8. Entraînement de la Régression Logistique (Maison)...")
    model_lr = MaRegressionLogistique(taux_apprentissage=0.1, iterations=100)
    model_lr.fit(X_train, y_train.values)
    pred_lr = model_lr.predict(X_test)
    
    acc_lr = accuracy_score(y_test, pred_lr)
    f1_lr = f1_score(y_test, pred_lr)
    resultats_finaux['LR (Maison) Acc'] = acc_lr
    resultats_finaux['LR (Maison) F1'] = f1_lr
    print(f"  Accuracy : {acc_lr:.2%} | F1-Score : {f1_lr:.2%}")

    # 9. COMPARAISON AVEC SCIKIT-LEARN
    print("\n9. Calcul de la comparaison avec Scikit-Learn...")
    resultats_pro = entrainer_modeles_pro(X_train, y_train, X_test, y_test)
    resultats_finaux.update(resultats_pro)

    # 10. VISUALISATION
    print("\n10. Génération des graphiques...")
    afficher_histogramme(scores_nb, titre="Scores de Toxicite - Naive Bayes")
    comparer_modeles(resultats_finaux)

    print("\n=== TERMINÉ ===")
    print("Points respectés : EDA, Missing Values, Balancing, Comparison, Metrics.")

if __name__ == "__main__":
    main()
