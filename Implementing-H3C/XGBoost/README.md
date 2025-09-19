# XGBoost pour le Scoring de Crédit

## 📌 Description
XGBoost est un modèle d’ensemble basé sur le **Gradient Boosting**.  
Il est efficace pour capturer des interactions complexes entre variables et gère bien les données déséquilibrées.

## ⚙️ Variantes
- **CLASSIC** : XGBoost standard
- **CS (Cost-Sensitive)** : pondération par coût de misclassification (MCI)
- **H3C** : pondération par coûts métier et stratégique

## 📦 Dépendances
```bash
pip install xgboost scikit-learn pandas numpy
```

## ▶️ Exemple d’utilisation
```python
from xgboost import XGBClassifier
model = XGBClassifier(scale_pos_weight=ratio)
model.fit(X_train, y_train)
```

## ✅ Résultats attendus
- Bonne performance sur les datasets déséquilibrés
- Explicabilité possible via SHAP values
- Export des résultats en Excel et Word

## 🔎 Avantages
- Performant et robuste
- Gère le déséquilibre avec `scale_pos_weight`
- Explicabilité avec SHAP

## ⚠️ Limites
- Plus complexe à paramétrer
- Temps d’entraînement plus élevé
