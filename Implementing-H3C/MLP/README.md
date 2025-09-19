# MLP (Multilayer Perceptron) pour le Scoring de Crédit

## 📌 Description
Le MLP est un réseau de neurones artificiels capable de modéliser des relations complexes entre les variables.

## ⚙️ Variantes
- **CLASSIC** : MLP standard
- **CS (Cost-Sensitive)** : pondération des classes par MCI
- **H3C** : pondération par coûts métier

## 📦 Dépendances
```bash
pip install scikit-learn pandas numpy
```

## ▶️ Exemple d’utilisation
```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500)
model.fit(X_train, y_train)
```

## ✅ Résultats attendus
- Capacité à capturer des relations non-linéaires
- Performance améliorée avec tuning

## 🔎 Avantages
- Flexible
- Peut modéliser des interactions complexes

## ⚠️ Limites
- Moins interprétable
- Paramétrage délicat
- Temps d’entraînement plus long
