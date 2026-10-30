"""Projet 2 - Prediction avec le modele entraine.

Usage : python src/predict.py chemin/vers/flux.csv
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_model = None


def _charger_modele():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(df: pd.DataFrame) -> list:
    """df : memes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    X = df.drop(columns=["attaque"], errors="ignore")
    return [int(p) for p in _charger_modele().predict(X)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/predict.py flux.csv")
    print(predict(pd.read_csv(sys.argv[1])))
