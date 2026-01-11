import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import preparer_donnees, equilibrer_donnees
from src.vectorization import SimpleBagOfWords
from src.naive_bayes import MonNaiveBayes
from src.logistic_regression import MaRegressionLogistique
from src.visualization import (afficher_distribution_classes, afficher_matrice_confusion,
                                afficher_histogramme, comparer_modeles)
from sklearn_models import entrainer_modeles_pro

def afficher_metriques(nom, y_true, y_pred):
    """
    Affiche toutes les métriques importantes pour évaluer un modèle.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n  📈 Résultats : {nom}")
    print(f"     Accuracy  : {acc:.2%}  (Taux de bonnes prédictions)")
    print(f"     Précision : {prec:.2%}  (Éviter les faux positifs)")
    print(f"     Rappel    : {rec:.2%}  (Détecter tous les vrais toxiques)")
    print(f"     F1-Score  : {f1:.2%}  (Équilibre Précision/Rappel)")
    
    return {'Acc': acc, 'Prec': prec, 'Rec': rec, 'F1': f1}

def discuter_resultats(resultats, y_test):
    """
    ÉTAPE 7 : DISCUSSION DES RÉSULTATS
    Analyse comparative des performances et recommandations.
    """
    print("\n" + "="*70)
    print("📊 DISCUSSION DES RÉSULTATS".center(70))
    print("="*70)
    
    # 1. Comparaison Fait-Maison vs Scikit-Learn
    print("\n🔹 Comparaison : Nos Implémentations vs Scikit-Learn")
    print("-" * 70)
    
    nb_diff = abs(resultats['NB (Maison)']['F1'] - resultats['NB (Sklearn)']['F1'])
    lr_diff = abs(resultats['LR (Maison)']['F1'] - resultats['LR (Sklearn)']['F1'])
    
    print(f"\n   Naive Bayes :")
    print(f"      Notre implémentation : {resultats['NB (Maison)']['F1']:.2%}")
    print(f"      Scikit-Learn         : {resultats['NB (Sklearn)']['F1']:.2%}")
    print(f"      Différence           : {nb_diff:.2%}")
    
    print(f"\n   Régression Logistique :")
    print(f"      Notre implémentation : {resultats['LR (Maison)']['F1']:.2%}")
    print(f"      Scikit-Learn         : {resultats['LR (Sklearn)']['F1']:.2%}")
    print(f"      Différence           : {lr_diff:.2%}")
    
    if nb_diff < 0.03 and lr_diff < 0.03:
        print("\n   ✅ Excellent ! Nos implémentations sont aussi bonnes que Scikit-Learn !")
    elif nb_diff < 0.05 and lr_diff < 0.05:
        print("\n   ✅ Très bien ! Différence acceptable (<5%)")
    else:
        print("\n   ⚠️  Scikit-Learn performe mieux (optimisations internes)")
    
    # 2. Quel modèle est le meilleur ?
    print("\n🔹 Meilleur Modèle")
    print("-" * 70)
    
    meilleur = max(resultats.items(), key=lambda x: x[1]['F1'])
    print(f"\n   🏆 Champion : {meilleur[0]}")
    print(f"      F1-Score : {meilleur[1]['F1']:.2%}")
    
    if 'NB' in meilleur[0]:
        print("\n   💡 Naive Bayes gagne ! Pourquoi ?")
        print("      • Hypothèse d'indépendance des mots fonctionne bien ici")
        print("      • Simple et efficace pour la classification de texte")
    else:
        print("\n   💡 Régression Logistique gagne ! Pourquoi ?")
        print("      • Capture mieux les relations entre mots")
        print("      • Plus flexible que Naive Bayes")
    
    # 3. Analyse de l'équilibre
    print("\n🔹 Équilibre du Jeu de Test")
    print("-" * 70)
    
    nb_toxiques = sum(y_test)
    nb_sains = len(y_test) - nb_toxiques
    ratio_tox = nb_toxiques / len(y_test)
    
    print(f"\n   Toxiques : {nb_toxiques} ({ratio_tox*100:.1f}%)")
    print(f"   Sains    : {nb_sains} ({(1-ratio_tox)*100:.1f}%)")
    
    if 0.4 <= ratio_tox <= 0.6:
        print("\n   ✅ Classes parfaitement équilibrées (40-60%) !")
        print("      • Les modèles apprennent équitablement")
        print("      • Accuracy et F1-Score sont fiables")
    else:
        print("\n   ⚠️  Déséquilibre détecté")
        print("      • Privilégier le F1-Score plutôt que l'Accuracy")
    
    # 4. Recommandations
    print("\n🔹 Pistes d'Amélioration")
    print("-" * 70)
    
    print("\n   🚀 Pour aller plus loin :")
    print("      1. Utiliser TF-IDF au lieu de Bag of Words")
    print("      2. Augmenter le vocabulaire (2000-5000 mots)")
    print("      3. Tester des n-grams (bi-grams : 'très mauvais')")
    print("      4. Essayer d'autres modèles (SVM, Random Forest)")
    print("      5. Utiliser des embeddings (Word2Vec, BERT)")
    
    print("\n   📚 Concepts à approfondir :")
    print("      • Validation croisée (K-Fold)")
    print("      • Régularisation (L1, L2)")
    print("      • Feature engineering avancé")
    print("      • Deep Learning (LSTM, Transformers)")
    
    print("\n" + "="*70 + "\n")

def main():
    print("="*70)
    print(" PROJET ML : DÉTECTION DE TOXICITÉ ".center(70))
    print(" Version Académique Complète ".center(70))
    print("="*70 + "\n")

    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    print("📂 ÉTAPE 1 : Chargement des données...")
    try:
        df = pd.read_csv('data/archive/train.csv')
        df = df.sample(n=10000, random_state=42)
        print(f"   ✅ {len(df)} commentaires chargés")
    except FileNotFoundError:
        print("   ❌ Erreur : Fichier 'data/archive/train.csv' introuvable")
        return

    # ÉTAPE 2 : PRÉTRAITEMENT + MISSING VALUES
    print("\n🧹 ÉTAPE 2 : Prétraitement et gestion des valeurs manquantes...")
    df = preparer_donnees(df)
    print("   ✅ Textes nettoyés et valeurs manquantes traitées")

    # ÉTAPE 3 : ANALYSE EXPLORATOIRE (EDA)
    print("\n📊 ÉTAPE 3 : Analyse exploratoire (EDA)...")
    afficher_distribution_classes(df)
    print("   ✅ Graphique de distribution généré")

    # ÉTAPE 4 : GESTION DU DÉSÉQUILIBRE
    print("\n⚖️  ÉTAPE 4 : Gestion du déséquilibre des classes...")
    df = equilibrer_donnees(df)
    print(f"   ✅ Dataset équilibré : {len(df)} lignes")

    # ÉTAPE 5 : SÉPARATION TRAIN/TEST
    print("\n✂️  ÉTAPE 5 : Séparation Train/Test (80/20)...")
    X_train_brut, X_test_brut, y_train, y_test = train_test_split(
        df['texte_nettoye'], df['est_toxique'], test_size=0.2, random_state=42
    )
    print(f"   ✅ Train : {len(X_train_brut)} | Test : {len(X_test_brut)}")

    # ÉTAPE 6 : VECTORISATION + NORMALISATION
    print("\n🔢 ÉTAPE 6 : Vectorisation (Bag of Words) + Normalisation...")
    bow = SimpleBagOfWords(max_mots=1000, normaliser=True)
    X_train = bow.fit_transform(X_train_brut)
    X_test = bow.transform(X_test_brut)
    print("   ✅ Textes transformés en matrices numériques normalisées")

    # ÉTAPE 7 : ENTRAÎNEMENT DES MODÈLES
    print("\n🤖 ÉTAPE 7 : Entraînement et Comparaison des Modèles")
    print("="*70)
    
    resultats = {}

    # --- Naive Bayes (Fait-Maison) ---
    print("\n🔷 Naive Bayes (Implémentation Maison)")
    print("-" * 70)
    model_nb = MonNaiveBayes()
    model_nb.fit(X_train, y_train.values)
    pred_nb = model_nb.predict(X_test)
    scores_nb = model_nb.predict_proba(X_test)
    resultats['NB (Maison)'] = afficher_metriques("Naive Bayes (Maison)", y_test, pred_nb)
    afficher_matrice_confusion(y_test, pred_nb, "NaiveBayes_Maison")
    print("   ✅ Matrice de confusion sauvegardée")

    # --- Régression Logistique (Fait-Maison) ---
    print("\n🔷 Régression Logistique (Implémentation Maison)")
    print("-" * 70)
    model_lr = MaRegressionLogistique(taux_apprentissage=0.01, iterations=200)
    model_lr.fit(X_train, y_train.values)
    pred_lr = model_lr.predict(X_test)
    scores_lr = model_lr.predict_proba(X_test)
    resultats['LR (Maison)'] = afficher_metriques("Régression Logistique (Maison)", y_test, pred_lr)
    afficher_matrice_confusion(y_test, pred_lr, "LogReg_Maison")
    print("   ✅ Matrice de confusion sauvegardée")

    # --- Comparaison avec Scikit-Learn ---
    print("\n🔷 Modèles Scikit-Learn (Référence)")
    print("-" * 70)
    resultats_sklearn = entrainer_modeles_pro(X_train, y_train, X_test, y_test)
    
    resultats['NB (Sklearn)'] = {
        'Acc': resultats_sklearn['NB (Sklearn) Acc'],
        'F1': resultats_sklearn['NB (Sklearn) F1'],
        'Prec': 0,  # Non calculé pour simplifier
        'Rec': 0
    }
    resultats['LR (Sklearn)'] = {
        'Acc': resultats_sklearn['LR (Sklearn) Acc'],
        'F1': resultats_sklearn['LR (Sklearn) F1'],
        'Prec': 0,
        'Rec': 0
    }
    
    print(f"\n  📈 Naive Bayes (Sklearn)")
    print(f"     Accuracy : {resultats['NB (Sklearn)']['Acc']:.2%}")
    print(f"     F1-Score : {resultats['NB (Sklearn)']['F1']:.2%}")
    
    print(f"\n  📈 Régression Logistique (Sklearn)")
    print(f"     Accuracy : {resultats['LR (Sklearn)']['Acc']:.2%}")
    print(f"     F1-Score : {resultats['LR (Sklearn)']['F1']:.2%}")

    # ÉTAPE 8 : VISUALISATIONS
    print("\n📊 ÉTAPE 8 : Génération des Visualisations...")
    afficher_histogramme(scores_nb, "Scores_NaiveBayes")
    afficher_histogramme(scores_lr, "Scores_RegressionLogistique")
    
    # Graphique de comparaison (F1-Scores uniquement)
    f1_scores = {k: v['F1'] for k, v in resultats.items()}
    comparer_modeles(f1_scores)
    print("   ✅ Tous les graphiques générés")

    # ÉTAPE 9 : DISCUSSION DES RÉSULTATS
    print("\n💬 ÉTAPE 9 : Discussion et Analyse")
    discuter_resultats(resultats, y_test)

    # FIN
    print("="*70)
    print(" PROJET TERMINÉ AVEC SUCCÈS ! ".center(70))
    print("="*70)
    print("\n✅ Tous les fichiers générés :")
    print("   • eda_distribution.png")
    print("   • matrice_NaiveBayes_Maison.png")
    print("   • matrice_LogReg_Maison.png")
    print("   • scores_naivebayes.png")
    print("   • scores_regressionlogistique.png")
    print("   • comparaison_modeles.png")
    print("\n📚 Vérifiez les graphiques pour une analyse visuelle complète !\n")

if __name__ == "__main__":
    main()