# Naïve Bayes pour le Scoring de Crédit

## 📌 Description
Naïve Bayes repose sur le théorème de Bayes avec une hypothèse d’indépendance entre les variables.  
C’est un modèle simple, rapide et efficace pour les données catégorielles.

## ⚙️ Variantes
- **CLASSIC** : Naïve Bayes standard
- **CS (Cost-Sensitive)** : pondération par coût de misclassification
- **H3C** : pondération par coûts métier

## 📦 Dépendances
```bash
pip install scikit-learn pandas numpy
```

## ▶️ Exemple d’utilisation
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
```

## ✅ Résultats attendus
- Rapide à entraîner
- Utile comme baseline

## 🔎 Avantages
- Simplicité
- Efficace sur petits datasets

## ⚠️ Limites
- Hypothèse d’indépendance souvent irréaliste
- Moins performant sur variables corrélées
