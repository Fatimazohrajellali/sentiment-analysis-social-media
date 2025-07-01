# 💬 Analyse de Sentiments sur les Réseaux Sociaux

---

## 📌 Objectif du projet

Ce projet vise à **extraire**, **nettoyer**, **analyser** et **classifier** les commentaires issus d’Instagram et de YouTube selon leur tonalité (positive, négative ou neutre) en utilisant des techniques avancées de traitement automatique du langage naturel (TALN) et de machine learning. Une application **Streamlit** permet une visualisation et une interaction simples avec les résultats.

---

## 🔍 Étapes du projet

1. **Collecte des données**  
   - *Instagram* : Utilisation de Selenium pour récupérer les commentaires.  
   - *YouTube* : Utilisation de l'API YouTube pour extraire les commentaires publics.

2. **Prétraitement des données**  
   - Nettoyage des textes (ponctuation, stop words, emojis…)  
   - Tokenisation et normalisation

3. **Étiquetage automatique des sentiments**  
   - Utilisation de `distilbert-base-uncased-finetuned-sst-2-english` (via HuggingFace) pour labelliser automatiquement les commentaires

4. **Entraînement des modèles de Machine Learning**  
   Modèles testés :  
   - Régression Logistique Multinomiale  
   - SVM  
   - Naive Bayes  
   - Random Forest  
   - Decision Tree  
   - KNN  

   Métriques utilisées :  
   - Accuracy  
   - F1-score  
   - Matrice de confusion  
   - ROC-AUC Score  

5. **Déploiement avec Streamlit**  
   Application simple et interactive pour permettre aux utilisateurs d’explorer les résultats.

6. **Technologies utilisées**  
   - Python  
   - Pandas, Scikit-learn, NLTK  
   - HuggingFace Transformers (DistilBERT)  
   - Streamlit  
   - Selenium  
   - YouTube Data API v3  

---

## 🎯 Résultats et comparaison des modèles

| Modèle                         | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) | ROC-AUC Score | Remarques                          |
|-------------------------------|----------|---------------------|-------------------|---------------------|---------------|-----------------------------------|
| **Régression Logistique**      | 0.90     | 0.91                | 0.90              | 0.90                | 0.9590        | Modèle optimisé, très performant  |
| **SVM**                       | 0.90     | 0.90                | 0.90              | 0.90                | 0.9773        | Meilleur AUC, bon équilibre        |
| **Random Forest**              | 0.91     | 0.91                | 0.91              | 0.91                | N/A           | Meilleurs hyperparamètres utilisés |
| **Decision Tree**              | 0.88     | 0.88                | 0.88              | 0.87                | 0.8916        | Performant mais moins stable       |
| **Naive Bayes**                | 0.79     | 0.80                | 0.79              | 0.79                | 0.9205        | Bon pour un modèle simple          |
| **KNN**                       | 0.58     | 0.74                | 0.58              | 0.57                | 0.8946        | Moins performant sur ce dataset    |
| **DistilBERT (Transformer)**  | 0.86     | 0.86                | 0.86              | 0.85                | N/A           | Performances solides en NLP        |
| **XLNet (Transformer)**       | 0.84     | 0.84                | 0.84              | 0.84                | N/A           | Bon modèle, légèrement en-dessous  |

---

## 📧 Contact

**Auteure** : Fatima-Zohra Jellali  
📍 Rabat, Maroc  
✉️ fatimazohrajellali@gmail.com
