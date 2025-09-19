# Credit Scoring - Comparaison de Modèles

## 📌 Description
Ce projet implémente et compare plusieurs méthodes de scoring de crédit appliquées à des datasets CSV.  
Chaque méthode est déclinée en trois variantes :
- **CLASSIC** : modèle standard sans pondération
- **CS (Cost-Sensitive)** : modèle pondéré pour minimiser les coûts de misclassification (MCI)
- **H3C** : modèle pondéré intégrant les coûts métier, stratégique et d’interprétabilité

## 📊 Méthodes disponibles
- [XGBoost](./xgboost/README.md)
- [Régression Logistique](./logistic_regression/README.md)
- [SVM](./svm/README.md)
- [Naïve Bayes](./naive_bayes/README.md)
- [MLP (Multilayer Perceptron)](./mlp/README.md)

## ⚙️ Fonctionnalités communes
- Chargement de datasets CSV
- Harmonisation des colonnes (`montant`, `taux`, `durée`, `cible`)
- Standardisation de la cible (`0=IMPAYE`, `1=PAYE`)
- Encodage des variables
- Entraînement et évaluation des modèles
- Export des résultats en **Excel**, **CSV** et **Word**

## 📦 Dépendances
- Python 3.10+
- Bibliothèques :
  - pandas, numpy, scikit-learn
  - xgboost (pour XGBoost uniquement)
  - python-docx, openpyxl

Installez avec :
```bash
pip install -r requirements.txt
```

## 📁 Organisation
Chaque dossier contient un `README.md` avec :
- description spécifique
- dépendances
- exemples d’utilisation
- résultats attendus
- points forts/faiblesses du modèle

## 📜 Licence
MIT License. Utilisation libre, mention appréciée.
