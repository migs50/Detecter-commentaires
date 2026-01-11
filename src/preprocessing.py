import pandas as pd
import re
import numpy as np

# ÉTAPE 1 : NETTOYAGE DES TEXTES

def nettoyer_texte(texte):
    """
    Simplifie le texte : minuscules, sans ponctuation, sans chiffres.
    """
    if not isinstance(texte, str):
        return ""
    
    texte = texte.lower()
    texte = re.sub(r'[^a-z\s]', '', texte)
    mots = texte.split()
    
    return " ".join(mots)

# ÉTAPE 2 : MISSING VALUES

def preparer_donnees(df, colonne_texte='comment_text'):
    """
    Nettoie les données et gère les valeurs manquantes.
    """
    print("📝 Nettoyage des commentaires...")
    
    df_propre = df.copy()
    
    # Remplacer les valeurs manquantes par du texte vide
    df_propre[colonne_texte] = df_propre[colonne_texte].fillna("")
    
    # Nettoyer chaque commentaire
    df_propre['texte_nettoye'] = df_propre[colonne_texte].apply(nettoyer_texte)
    
    # Créer la colonne cible
    colonnes_toxicite = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_propre['est_toxique'] = df_propre[colonnes_toxicite].max(axis=1)
    
    return df_propre

# ÉTAPE 4 : GESTION DU DÉSÉQUILIBRE

def equilibrer_donnees(df, colonne='est_toxique'):
    """
    Équilibre les classes avec Undersampling (méthode simple).
    """
    print("⚖️  Équilibrage des classes...")
    
    df_toxique = df[df[colonne] == 1]
    df_sain = df[df[colonne] == 0]
    
    # Prendre autant de sains que de toxiques
    n_echantillons = min(len(df_toxique), len(df_sain))
    df_sain_reduit = df_sain.sample(n=n_echantillons, random_state=42)
    df_tox_reduit = df_toxique.sample(n=n_echantillons, random_state=42)
    
    df_equilibre = pd.concat([df_tox_reduit, df_sain_reduit])
    return df_equilibre.sample(frac=1, random_state=42)