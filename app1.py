import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib
from wordcloud import WordCloud  # Import pour le Word Cloud

# Télécharger les ressources NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Charger le modèle entraîné
model_path = "C:/Users/INKA/Desktop/application_analyse_de_sentiment/best_rf_model_multipleavecgrid.pkl"
model = joblib.load(model_path)

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/INKA/Desktop/application_analyse_de_sentiment/data_label.csv')
    return df

df = load_data()

# Prétraitement du texte
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Supprimer les stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatisation
    return text

# Appliquer le prétraitement sur la colonne 'comment_cleaned'
df['processed_text'] = df['comment_cleaned'].apply(preprocess_text)

# Vectorisation TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiter à 5000 features pour l'efficacité
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

# Encodage des sentiments
sentiment_categories = [['negative', 'neutral', 'positive']]
ordinal_encoder = OrdinalEncoder(categories=sentiment_categories)
y_encoded = ordinal_encoder.fit_transform(df[['sentiment']])
y = y_encoded.flatten()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = joblib.load(model_path)

# Titre de l'application
st.title("Analyse de Sentiment - Visualisations")

# Sidebar pour les options
st.sidebar.header("Options de Visualisation")
visualization_option = st.sidebar.selectbox(
    "Choisissez une visualisation",
    ["Word Cloud", "Bar Plot des Mots les Plus Fréquents", "Interactive Sunburst Charts", "Correlation Heatmaps", "Distribution Plots"]
)

# Section pour analyser un texte
st.sidebar.header("Analyse de Texte")
user_input = st.sidebar.text_area("Entrez votre texte ici")

if st.sidebar.button("Analyser le Sentiment"):
    if user_input:
        # Prétraitement du texte saisi par l'utilisateur
        user_input_cleaned = preprocess_text(user_input)
        
        # Vectorisation du texte saisi
        user_input_tfidf = tfidf_vectorizer.transform([user_input_cleaned])
        
        # Prédiction du sentiment
        prediction = model.predict(user_input_tfidf)
        sentiment = ordinal_encoder.inverse_transform([prediction])[0][0]
        
        # Affichage du sentiment prédit
        st.sidebar.success(f"Le sentiment prédit est : **{sentiment}**")

        # Sauvegarder le sentiment prédit dans st.session_state
        if "predicted_sentiments" not in st.session_state:
            st.session_state["predicted_sentiments"] = []
        st.session_state["predicted_sentiments"].append(sentiment)

    else:
        st.sidebar.error("Veuillez entrer un texte pour l'analyse.")


# Word Cloud des Mots les Plus Fréquents par Sentiment
# Word Cloud des Mots les Plus Fréquents par Sentiment
if visualization_option == "Word Cloud":
    st.header("Nuage de Mots par Sentiment")
    
    # Option pour choisir entre le dataset ou le texte entré
    source_option = st.radio(
        "Source du texte pour le Word Cloud",
        ["Dataset complet", "Analyse de votre propre texte"],
        index=0
    )
    
    if source_option == "Analyse de votre propre texte":
        if user_input:
            # Générer le nuage de mots à partir du texte entré
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
            st.image(wordcloud.to_array(), use_column_width=True)
            st.caption("Nuage de mots généré à partir du texte que vous avez entré")
        else:
            st.warning("Veuillez d'abord entrer un texte dans la zone de texte à gauche")
    else:
        # Sélectionner le sentiment à visualiser (version originale)
        selected_sentiment = st.selectbox("Choisissez un sentiment", ["positive", "neutral", "negative"])
        
        # Filtrer les commentaires par sentiment
        filtered_text = " ".join(df[df['sentiment'] == selected_sentiment]['comment_cleaned'])
        
        # Générer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
        
        # Afficher le nuage de mots
        st.image(wordcloud.to_array(), use_column_width=True)
        st.caption(f"Nuage de mots pour les commentaires '{selected_sentiment}' du dataset")


# Bar Plot des Mots les Plus Fréquents par Sentiment
elif visualization_option == "Bar Plot des Mots les Plus Fréquents":
    st.header("Mots les Plus Fréquents par Sentiment")

    # Sélectionner le sentiment à visualiser
    selected_sentiment = st.selectbox("Choisissez un sentiment", ["positive", "neutral", "negative"])

    # Filtrer les commentaires par sentiment
    filtered_text = df[df['sentiment'] == selected_sentiment]['comment_cleaned']

    # Vectorisation des mots
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(filtered_text)

    # Obtenir les mots les plus importants
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = X_tfidf.sum(axis=0).A1
    word_importance = pd.DataFrame({'word': feature_names, 'score': tfidf_scores})
    word_importance = word_importance.sort_values(by='score', ascending=False).head(20)

    # Créer un bar plot
    fig = px.bar(word_importance, x='score', y='word', orientation='h', 
                 title=f"Top 20 des Mots les Plus Importants pour le Sentiment {selected_sentiment.capitalize()}")
    st.plotly_chart(fig)


# Interactive Sunburst Charts
elif visualization_option == "Interactive Sunburst Charts":
    st.header("Répartition des Sentiments")

    # Compter les occurrences de chaque sentiment
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["category", "value"]
    sentiment_counts["parent"] = ""  # Pas de parent car ce sont les catégories principales

    # Création du graphique Sunburst
    fig = px.sunburst(sentiment_counts, path=['category', 'parent'], values='value', 
                      color='category', color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"})

    # Affichage dans Streamlit
    st.plotly_chart(fig)


# Correlation Heatmaps
elif visualization_option == "Correlation Heatmaps":
    st.header("Heatmap des Co-Occurrences des Sentiments")

    # Création d'une matrice de co-occurrence des sentiments
    sentiment_counts = df['sentiment'].value_counts()
    co_occurrence_matrix = pd.DataFrame(0, index=sentiment_counts.index, columns=sentiment_counts.index)

    for i in sentiment_counts.index:
        for j in sentiment_counts.index:
            co_occurrence_matrix.loc[i, j] = (df['sentiment'] == i).sum() * (df['sentiment'] == j).sum()

    # Création de la heatmap avec Seaborn
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap="coolwarm", fmt="d", ax=ax)
    st.pyplot(fig)


# Distribution Plots
elif visualization_option == "Distribution Plots":
    st.header("Répartition des Sentiments sur les Nouveaux Commentaires")

    # Vérifier si des commentaires ont été testés
    if "predicted_sentiments" in st.session_state and len(st.session_state["predicted_sentiments"]) > 0:
        # Récupérer les sentiments prédites
        df_distribution = pd.DataFrame({'sentiment': st.session_state["predicted_sentiments"]})

        # Créer l'histogramme basé sur les résultats du modèle
        fig = px.histogram(df_distribution, x='sentiment', color='sentiment', text_auto=True)
        st.plotly_chart(fig)
    else:
        st.warning("Aucun commentaire testé pour le moment. Veuillez tester des commentaires pour voir la répartition des sentiments.")