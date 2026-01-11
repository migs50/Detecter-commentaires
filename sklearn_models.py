from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, f1_score

# Pourquoi utiliser Scikit-Learn ? 
# Pour vérifier si notre code "fait maison" fonctionne aussi bien que les outils pros.

def entrainer_modeles_pro(X_train, y_train, X_test, y_test):
    """
    Entraîne les versions Scikit-Learn et retourne leurs métriques.
    """
    # 1. Naive Bayes Pro
    nb_pro = MultinomialNB()
    nb_pro.fit(X_train, y_train)
    pred_nb = nb_pro.predict(X_test)
    acc_nb = accuracy_score(y_test, pred_nb)
    f1_nb = f1_score(y_test, pred_nb)
    
    # 2. Régression Logistique Pro
    lr_pro = SklearnLR(max_iter=1000)
    lr_pro.fit(X_train, y_train)
    pred_lr = lr_pro.predict(X_test)
    acc_lr = accuracy_score(y_test, pred_lr)
    f1_lr = f1_score(y_test, pred_lr)
    
    return {
        'NB (Sklearn) Acc': acc_nb,
        'NB (Sklearn) F1': f1_nb,
        'LR (Sklearn) Acc': acc_lr,
        'LR (Sklearn) F1': f1_lr
    }