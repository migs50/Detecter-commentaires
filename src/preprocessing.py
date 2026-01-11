import pandas as pd
import re

# ÉTAPE 1 : LE PRÉTRAITEMENT (Nettoyage)
# Pourquoi cette étape ? 
# Les commentaires internet sont souvent "sales" (mélange de majuscules/minuscules, ponctuation, smileys).
# Si on ne nettoie pas, l'ordinateur verra "Bonjour!" et "bonjour" comme deux mots différents.
# On veut simplifier le texte pour ne garder que l'essentiel : les mots.

def nettoyer_texte(texte):
    """
    Cette fonction prend un commentaire et le simplifie :
    1. Tout en minuscules (pour que 'Bonjour' et 'bonjour' soient identiques)
    2. Enlever tout ce qui n'est pas une lettre (chiffres, ponctuation)
    3. Diviser le texte en une liste de mots
    """
    if not isinstance(texte, str):
        return ""
    
    # 1. Minuscules
    texte = texte.lower()
    
    # 2. On garde seulement les lettres de a à z (et les espaces)
    # On utilise une "expression régulière" simple
    texte = re.sub(r'[^a-z\s]', '', texte)
    
    # 3. On découpe en mots (tokenization)
    mots = texte.split()
    
    # On rejoint les mots avec un espace pour avoir une phrase "propre"
    return " ".join(mots)

def preparer_donnees(df, colonne_texte='comment_text'):
    """
    Applique le nettoyage sur tout le tableau (DataFrame).
    """
    print("Nettoyage des commentaires en cours...")
    
    # Copie du tableau pour ne pas abîmer l'original
    df_propre = df.copy()
    
    # Remplacer les valeurs vides par du texte vide
    df_propre[colonne_texte] = df_propre[colonne_texte].fillna("")
    
    # Nettoyer chaque ligne
    df_propre['texte_nettoye'] = df_propre[colonne_texte].apply(nettoyer_texte)
    
    # Créer une colonne cible simple : 1 si toxique, 0 sinon
    # Le dataset Kaggle a plusieurs colonnes (toxic, severe_toxic, etc.)
    # On dit que si l'une d'elles est à 1, alors le commentaire est 'toxique'.
    colonnes_toxicite = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_propre['est_toxique'] = df_propre[colonnes_toxicite].max(axis=1)
    
    return df_propre

def equilibrer_donnees(df, colonne='est_toxique'):
    """
    ÉTAPE : GESTION DU DÉSÉQUILIBRE DES CLASSES
    Dans le monde réel, il y a beaucoup plus de commentaires "sains" que "toxiques".
    Pour que le modèle apprenne bien à reconnaître les toxiques, on réduit 
    le nombre de commentaires sains pour qu'il y en ait autant que de toxiques.
    C'est ce qu'on appelle le "Undersampling".
    """
    print("Équilibrage des classes (Undersampling)...")
    df_toxique = df[df[colonne] == 1]
    df_sain = df[df[colonne] == 0]
    
    # On prend autant de sains que de toxiques
    n_echantillons = min(len(df_toxique), len(df_sain))
    df_sain_reduit = df_sain.sample(n=n_echantillons, random_state=42)
    df_tox_reduit = df_toxique.sample(n=n_echantillons, random_state=42)
    
    # On mélange les deux
    df_equilibre = pd.concat([df_tox_reduit, df_sain_reduit])
    return df_equilibre.sample(frac=1, random_state=42) # Mélange final
