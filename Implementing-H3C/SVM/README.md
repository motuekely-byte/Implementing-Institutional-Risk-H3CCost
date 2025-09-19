# SVM (Support Vector Machine) pour le Scoring de Crédit

## 📌 Description
Les SVM sont des modèles puissants pour la classification, surtout efficaces dans les espaces de grande dimension.

## ⚙️ Variantes
- **CLASSIC** : SVM standard
- **CS (Cost-Sensitive)** : pondération par coût de misclassification
- **H3C** : pondération par coûts métier

## 📦 Dépendances
```bash
pip install scikit-learn pandas numpy
```

## ▶️ Exemple d’utilisation
```python
from sklearn.svm import SVC
model = SVC(class_weight="balanced")
model.fit(X_train, y_train)
```

## ✅ Résultats attendus
- Bonne séparation des classes
- Support natif pour `class_weight`

## 🔎 Avantages
- Efficace sur données non linéaires
- Gère le déséquilibre via `class_weight`

## ⚠️ Limites
- Coût computationnel élevé sur grands datasets
- Moins interprétable
