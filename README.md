# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Auteure :** Adja Fatoumata Banna BARRY — adjafatoumatabanna@gmail.com

Un IDS qui classe chaque flux réseau en **normal (0)** ou **attaque (1)** à partir de `data/flux_train.csv`
(3 000 flux, ≈ 28 % d'attaques, 90 valeurs manquantes sur `duration`).

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py                    # entraîne, compare les modèles, écrit models/model.joblib
python src/predict.py data/flux_train.csv   # prédit (colonne attaque ignorée si présente)
```

## Structure
```
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py
└── models/model.joblib
```

## Résultats

**Nom / Prénom :** BARRY Adja Fatoumata Banna

Pipeline : imputation par la médiane + `StandardScaler` (variables numériques), `OneHotEncoder` (`protocol`),
puis modèle. Validation : 25 % des données (750 flux, découpage stratifié, `random_state=42`) ;
`class_weight="balanced"` pour tenir compte du déséquilibre des classes.

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 validation croisée (5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,919 | 0,899 | 0,940 | 0,896 |
| Arbre de décision (profondeur 6) | 0,959 | 0,957 | 0,961 | 0,935 |
| **Forêt aléatoire (300 arbres)** | **0,973** | **0,971** | **0,976** | **0,960** |

**Modèle retenu et pourquoi :** la forêt aléatoire, meilleure sur tous les indicateurs, à la fois sur le jeu de
validation et en validation croisée (plus fiable car moins dépendante d'un seul découpage). Elle capte mieux
les relations non linéaires entre `syn_rate`, `nb_conn_same_host`, `nb_failed_logins`, etc. Elle a été
ré-entraînée sur les 3 000 flux avant d'être enregistrée dans `models/model.joblib`.

**Analyse des erreurs (faux négatifs / faux positifs) :** matrice de confusion de la forêt aléatoire sur la
validation (750 flux) :

|  | Prédit normal | Prédit attaque |
|---|---|---|
| **Réel normal** | 537 | 5 (faux positifs) |
| **Réel attaque** | 6 (faux négatifs) | 202 |

Le modèle rate **6 attaques sur 208** (2,9 %). C'est le pire cas en sécurité : une attaque non détectée passe
inaperçue et peut compromettre le réseau (accès par force brute, déni de service), alors qu'un faux positif
coûte seulement une vérification de l'analyste (5 sur 542 flux normaux). La régression logistique en rate 21,
d'où le choix d'un modèle plus sensible. Pour réduire encore les faux négatifs, on pourrait abaisser le seuil de
décision de `predict_proba` au prix de quelques faux positifs en plus.
