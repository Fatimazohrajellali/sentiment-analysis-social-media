# sentiment-analysis-social-media
# 💬 Analyse de Sentiments sur les Réseaux Sociaux

## 📌 Objectif du projet
Ce projet vise à extraire, nettoyer, analyser et classifier les commentaires issus d’**Instagram** et de **YouTube** selon leur tonalité (**positive**, **négative** ou **neutre**) en utilisant des techniques de **traitement automatique du langage naturel (TALN)** et de **machine learning**. Une application **Streamlit** permet une visualisation et une interaction faciles avec les résultats.

---

## 🔍 Étapes du projet

### 1. Collecte des données
- **Instagram** : Utilisation de **Selenium** pour récupérer les commentaires.
- **YouTube** : Utilisation de l'**API YouTube** pour extraire les commentaires publics.

### 2. Prétraitement des données
- Nettoyage des textes (ponctuation, stop words, emojis…)
- Tokenisation, normalisation.

### 3. Étiquetage automatique des sentiments
Utilisation de `distilbert-base-uncased-finetuned-sst-2-english` (via HuggingFace) pour labelliser automatiquement les commentaires 
#### 4.Entraînement des modèles de Machine Learning 
Modèles testés :

Régression Logistique Multinomiale

SVM

Naive Bayes

Random Forest

Decision Tree

KNN

Métriques utilisées :

Accuracy

F1-score

Matrice de confusion

auc-scor 
##### 5.Déploiement avec Streamlit 
Application simple et interactive  
###### 6.Technologies utilisées
 Python

Pandas, Scikit-learn, NLTK

HuggingFace Transformers (DistilBERT)

Streamlit

Selenium

YouTube Data API v3 
####### 7.Résultats attendus 
Analyse automatique de sentiments sur plusieurs sources sociales

Interface interactive pour les utilisateurs

Bonne précision des modèles supervisés classiques

Auteure
Fatima-Zohra Jellali
📍 Rabat, Maroc
📧 fatimazohrajellali@gmail.com 
 
