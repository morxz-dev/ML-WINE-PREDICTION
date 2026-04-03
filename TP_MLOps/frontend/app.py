#Author: Mènéli Herve Adjole
import os
import requests
import pandas as pd
import json
import streamlit as st

st.set_page_config(page_title="Prédiction ML", layout="wide")
st.title("🧪 Prédiction Machine Learning")
st.markdown("Interface API MLOps")

# Sidebar pour choisir le mode d'entrée
st.sidebar.header("📥 Mode d'entrée")
input_mode = st.sidebar.radio("Choisir une option :", ["Manuel", "Depuis fichier"])

features_list = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids",
    "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines",
    "proline"
]

input_data = None

# --- Mode manuel ---
if input_mode == "Manuel":
    st.sidebar.header("📊 Features à prédire")
    features = {}
    for feature in features_list:
        features[feature] = st.sidebar.number_input(feature, value=0.0, step=0.1)
    input_data = pd.DataFrame([features])

# --- Mode fichier ---
else:
    uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV ou JSON", type=["csv", "json"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                input_data = pd.read_csv(uploaded_file)
            else:
                raw = json.load(uploaded_file)
                input_data = pd.DataFrame(raw["features"])
            
            st.subheader("Données chargées :")
            st.dataframe(input_data)

            missing_cols = set(features_list) - set(input_data.columns)
            if missing_cols:
                st.error(f"Colonnes manquantes : {missing_cols}")
            else:
                input_data = input_data[features_list]

        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")

# --- Bouton prédiction ---
API_URL = "https://ml-wine-prediction.onrender.com/predict"

def get_api_headers():
    """
    Retourne les headers pour éviter le 403 sur Render.
    Utilise User-Agent et clé API si définie dans les secrets.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json"
    }
    api_key = os.environ.get("RENDER_API_KEY")  # à définir si service privé
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

if st.sidebar.button("Prédire"):
    if input_data is not None and not input_data.empty:
        with st.spinner("Prédiction en cours..."):
            try:
                features_payload = input_data.to_dict(orient="records")
                headers = get_api_headers()
                response = requests.post(API_URL, json={"features": features_payload}, headers=headers, timeout=10)

                # Debug : afficher status code et contenu brut
                st.write(f"Status code API: {response.status_code}")
                st.write(f"Contenu brut API: {response.text[:500]}")  # limiter affichage à 500 caractères

                # Vérifier si la réponse est JSON
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    st.error("Réponse API invalide (non JSON). Voir contenu brut ci-dessus.")
                    st.stop()

                predictions = result.get("predictions", [])
                probas = result.get("proba", [])

                st.subheader("Résultats des prédictions")
                for i, row in input_data.iterrows():
                    st.markdown(f"### Ligne {i+1}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prédiction", predictions[i] if i < len(predictions) else "N/A")
                        if i < len(probas):
                            st.metric("Confiance", f"{max(probas[i].values()):.2%}")
                    with col2:
                        if i < len(probas):
                            st.json(probas[i])

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API: {e}")
    else:
        st.warning("Veuillez saisir des données ou charger un fichier avant de prédire.")

# Infos API
st.sidebar.markdown("---")
st.sidebar.info("**API Status:** https://ml-wine-prediction.onrender.com/health")
st.sidebar.info("**API Docs:** https://ml-wine-prediction.onrender.com/docs")
st.sidebar.info("**Author:** Mènéli Herve Adjole")
