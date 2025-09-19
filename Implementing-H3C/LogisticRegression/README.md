# Régression Logistique pour le Scoring de Crédit

## 📌 Description
La régression logistique est un modèle linéaire simple et interprétable, largement utilisé dans le scoring de crédit.

## ⚙️ Variantes
- **CLASSIC** : régression logistique standard
- **CS (Cost-Sensitive)** : pondération des classes par MCI
- **H3C** : pondération par coûts métier

## 📦 Dépendances
```bash
pip install scikit-learn pandas numpy
```

## ▶️ Exemple d’utilisation
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
```

## ✅ Résultats attendus
- Bon benchmark de comparaison
- Interprétation claire des coefficients
- Export des résultats en Excel et Word

## 🔎 Avantages
- Simplicité
- Interprétable
- Rapide à entraîner

## ⚠️ Limites
- Moins performant sur relations non-linéaires
- Sensible aux variables fortement corrélées
