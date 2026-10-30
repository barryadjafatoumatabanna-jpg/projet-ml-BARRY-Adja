"""Projet 2 - Entrainement d'un IDS (detection d'attaques reseau).

Usage : python src/train.py
Compare plusieurs modeles sur un jeu de validation, garde le meilleur (F1 de la
classe attaque), le re-entraine sur toutes les donnees et l'enregistre dans
models/model.joblib.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "flux_train.csv"
MODEL_PATH = BASE / "models" / "model.joblib"
SEED = 42

NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]


def construire_pipeline(modele) -> Pipeline:
    """Pretraitement (imputation, echelle, encodage) + modele dans un Pipeline."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", modele)])


def main() -> None:
    df = pd.read_csv(DATA)
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Repartition attaque :\n", df["attaque"].value_counts(normalize=True).round(3))

    X, y = df[NUM + CAT], df["attaque"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    modeles = {
        "Regression logistique": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Arbre de decision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Foret aleatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    print("\n=== Validation ===")
    for nom, m in modeles.items():
        pipe = construire_pipeline(m).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        f1 = f1_score(y_val, pred)
        cv_f1 = cross_val_score(construire_pipeline(m), X, y, cv=cv, scoring="f1").mean()
        scores[nom] = cv_f1
        print(f"\n--- {nom} ---")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print("Matrice de confusion [[VN FP], [FN VP]] :\n", confusion_matrix(y_val, pred))
        print(f"F1={f1:.3f} rappel={recall_score(y_val, pred):.3f} "
              f"precision={precision_score(y_val, pred):.3f} | F1 CV (5 plis)={cv_f1:.3f}")

    meilleur = max(scores, key=scores.get)
    print(f"\nModele retenu : {meilleur} (F1 CV = {scores[meilleur]:.3f})")

    final = construire_pipeline(modeles[meilleur]).fit(X, y)  # toutes les donnees
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print("Modele enregistre dans", MODEL_PATH)


if __name__ == "__main__":
    main()
