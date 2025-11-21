import streamlit as st
import numpy as np
import joblib
from deepface import DeepFace
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile

# Configuration pour le cloud
st.set_page_config(
    page_title="Reconnaissance Faciale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    h1, h2, h3 { color: #1a1a2e; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(90deg, #00bcd4, #0097a7);
        color: white; border: none; padding: 12px 30px;
        border-radius: 8px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Charge les modèles avec gestion d'erreurs améliorée"""
    try:
        classifier = joblib.load('face_classifier.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        st.success("Modèles chargés avec succès")
        return classifier, label_encoder
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None

@st.cache_resource
def load_database_embeddings():
    """Charge les embeddings avec gestion d'erreurs"""
    try:
        embeddings_db = np.load('embeddings_database.npy', allow_pickle=True).item()
        st.success("Base de données d'embeddings chargée")
        return embeddings_db
    except:
        st.warning("Base de données d'embeddings non disponible")
        return {}

def extract_embedding(image_path):
    """Extrait l'embedding facial"""
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name='ArcFace',
            enforce_detection=True,
            detector_backend='retinaface'
        )[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse du visage: {str(e)}")
        return None

def find_top_k_similar(embedding, embeddings_db, k=5):
    """Trouve les k visages les plus similaires"""
    if not embeddings_db:
        return [], []
    
    filenames = list(embeddings_db.keys())
    db_embeddings = np.array([embeddings_db[f] for f in filenames])
    
    similarities = cosine_similarity([embedding], db_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    top_files = [filenames[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    
    return top_files, top_scores

# Interface principale
st.markdown("# Reconnaissance Faciale")
st.markdown("*Système d'identification et d'analyse faciale*")

# Chargement des modèles
classifier, label_encoder = load_models()
embeddings_db = load_database_embeddings()

if classifier is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Paramètres")
    top_k = st.slider("Nombre de similarités", 1, 10, 3)
    confidence_threshold = st.slider("Seuil de confiance", 0, 100, 50)
    
    st.markdown("---")
    st.markdown("### Informations")
    st.markdown("""
    Technologies utilisées :
    - **ArcFace** pour l'extraction des caractéristiques
    - **SVM** pour la classification
    - **Similarité cosinus** pour les comparaisons
    """)

# Upload de fichier
uploaded_file = st.file_uploader(
    "Téléchargez une image contenant un visage",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Utilisation de fichiers temporaires pour le cloud
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Image téléchargée")
        img = Image.open(temp_path)
        st.image(img, use_column_width=True)
    
    with col2:
        if st.button("Analyser l'image", use_container_width=True):
            with st.spinner("Analyse du visage en cours..."):
                embedding = extract_embedding(temp_path)
            
            if embedding is not None:
                # Prédiction
                pred_encoded = classifier.predict([embedding])[0]
                pred_person = label_encoder.inverse_transform([pred_encoded])[0]
                prob = classifier.predict_proba([embedding])[0].max() * 100
                
                # Affichage des résultats
                st.markdown("### Résultats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Personne", pred_person)
                with col2:
                    st.metric("Confiance", f"{prob:.1f}%")
                
                # Similarités
                if embeddings_db:
                    top_files, top_scores = find_top_k_similar(embedding, embeddings_db, top_k)
                    if top_files:
                        st.markdown("#### Visages similaires")
                        for file, score in zip(top_files, top_scores):
                            st.write(f"- {file} : {score*100:.1f}%")
            
            # Nettoyage
            os.unlink(temp_path)