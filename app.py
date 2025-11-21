import streamlit as st
import numpy as np
import joblib
from deepface import DeepFace
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import requests
import gdown

# Configuration
st.set_page_config(
    page_title="Reconnaissance Faciale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #00bcd4;
    }
    
    .result-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        margin: 15px 0;
    }
    
    .confidence-bar {
        height: 8px;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
        border-radius: 10px;
        margin: 10px 0;
    }
    
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00bcd4, #0097a7);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(0, 188, 212, 0.4);
    }
    
    .download-progress {
        margin: 10px 0;
        padding: 10px;
        background: #e8f4f8;
        border-radius: 8px;
        border-left: 4px solid #00bcd4;
    }
</style>
""", unsafe_allow_html=True)

def download_classifier():
    """Télécharge le classifieur depuis Google Drive"""
    classifier_url = "https://drive.google.com/uc?id=1wctJ8RWAjfefz4IbtKEpr43Pk81sjvwd"
    classifier_path = "face_classifier.pkl"
    
    if not os.path.exists(classifier_path):
        with st.spinner("Téléchargement du classifieur en cours... Cette opération peut prendre quelques instants."):
            try:
                # Utilisation de gdown pour télécharger depuis Google Drive
                gdown.download(classifier_url, classifier_path, quiet=False)
                st.success("Classifieur téléchargé avec succès!")
                return True
            except Exception as e:
                st.error(f"Erreur lors du téléchargement du classifieur: {str(e)}")
                return False
    return True

@st.cache_resource
def load_models():
    """Charge les modèles en téléchargeant le classifieur si nécessaire"""
    # Vérifier et télécharger le classifieur si absent
    if not download_classifier():
        return None, None
    
    try:
        # Charger le classifieur
        classifier = joblib.load('face_classifier.pkl')
        
        # Charger le label encoder (supposé présent localement)
        if os.path.exists('label_encoder.pkl'):
            label_encoder = joblib.load('label_encoder.pkl')
        else:
            st.error("Fichier label_encoder.pkl non trouvé. Assurez-vous qu'il est présent dans le répertoire.")
            return None, None
            
        st.success("Modèles chargés avec succès")
        return classifier, label_encoder
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None

@st.cache_resource
def load_database_embeddings():
    """Charge les embeddings de la base de données"""
    try:
        if os.path.exists('embeddings_database.npy'):
            embeddings_db = np.load('embeddings_database.npy', allow_pickle=True).item()
            st.success("Base de données d'embeddings chargée")
            return embeddings_db
        else:
            st.warning("Base de données d'embeddings non disponible")
            return {}
    except Exception as e:
        st.error(f"Erreur lors du chargement des embeddings: {str(e)}")
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

st.markdown("---")

# Section d'information sur le téléchargement
with st.expander("ℹ️ Informations sur les modèles"):
    st.markdown("""
    **Configuration des modèles:**
    - 🧠 **Classifieur SVM**: Téléchargé automatiquement (85.4 MB)
    - 📝 **Label Encoder**: Fichier local
    - 🗄️ **Base d'embeddings**: Fichier local
    
    *Le premier lancement peut prendre quelques instants pour télécharger le classifieur.*
    """)

# Chargement des modèles
classifier, label_encoder = load_models()
embeddings_db = load_database_embeddings()

if classifier is None or label_encoder is None:
    st.error("Impossible de charger les modèles nécessaires. Vérifiez les fichiers requis.")
    st.stop()

# Sidebar pour les paramètres
with st.sidebar:
    st.markdown("### Paramètres d'analyse")
    top_k = st.slider("Nombre de visages similaires à afficher", 1, 10, 5)
    confidence_threshold = st.slider("Seuil de confiance minimum (%)", 0, 100, 30)
    
    st.markdown("---")
    st.markdown("### À propos du système")
    st.markdown("""
    Ce système utilise une technologie avancée de reconnaissance faciale basée sur :
    - **ArcFace** pour l'extraction des caractéristiques faciales
    - **SVM** pour la classification des visages
    - **Similarité cosinus** pour le regroupement des visages similaires
    
    Performance du modèle : 94.79% de précision
    """)
    
    # Information sur l'état des modèles
    st.markdown("---")
    st.markdown("### État des modèles")
    if os.path.exists('face_classifier.pkl'):
        file_size = os.path.getsize('face_classifier.pkl') / (1024 * 1024)
        st.success(f"✅ Classifieur: {file_size:.1f} MB")
    else:
        st.warning("⏳ Classifieur: En attente de téléchargement")
    
    if os.path.exists('label_encoder.pkl'):
        st.success("✅ Label Encoder: Chargé")
    else:
        st.error("❌ Label Encoder: Manquant")
    
    if embeddings_db:
        st.success(f"✅ Base d'embeddings: {len(embeddings_db)} visages")
    else:
        st.warning("⚠️ Base d'embeddings: Non disponible")

# Zone d'upload
st.markdown("### Télécharger une image pour analyse")

uploaded_file = st.file_uploader(
    "Sélectionnez une image au format JPG ou PNG",
    type=["jpg", "jpeg", "png"],
    help="Pour de meilleurs résultats, assurez-vous que le visage est bien éclairé et clairement visible"
)

if uploaded_file is not None:
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Affichage de l'image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Image sélectionnée")
        img = Image.open(temp_path)
        st.image(img, use_column_width=True)
    
    with col2:
        st.markdown("#### Analyse")
        if st.button("Lancer l'analyse faciale", use_container_width=True):
            with st.spinner("Extraction des caractéristiques faciales..."):
                embedding = extract_embedding(temp_path)
            
            if embedding is not None:
                # Prédiction
                with st.spinner("Classification en cours..."):
                    pred_encoded = classifier.predict([embedding])[0]
                    pred_person = label_encoder.inverse_transform([pred_encoded])[0]
                    prob = classifier.predict_proba([embedding])[0].max() * 100
                
                # Résultats principaux
                st.markdown("---")
                st.markdown("### Résultats de l'identification")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Personne identifiée", pred_person)
                with col2:
                    st.metric("Niveau de confiance", f"{prob:.2f}%")
                with col3:
                    status = "Identification fiable" if prob >= confidence_threshold else "Nécessite vérification"
                    st.metric("Statut", status)
                
                # Barre de confiance
                st.markdown("#### Niveau de confiance de l'identification")
                color = "#6bcf7f" if prob >= 80 else "#ffd93d" if prob >= 60 else "#ff6b6b"
                st.markdown(f"""
                <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="width: {min(prob, 100)}%; background: {color}; 
                    height: 20px; display: flex; align-items: center; justify-content: center;
                    color: white; font-weight: bold; font-size: 12px;">
                        {prob:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Recherche de visages similaires
                st.markdown("---")
                st.markdown(f"### Visages similaires dans la base de données")
                
                if embeddings_db:
                    top_files, top_scores = find_top_k_similar(embedding, embeddings_db, top_k)
                    
                    if top_files:
                        # Tableau des résultats
                        similarity_data = {
                            "Rang": list(range(1, len(top_files) + 1)),
                            "Fichier": top_files,
                            "Score de similarité": [f"{s*100:.2f}%" for s in top_scores]
                        }
                        
                        st.dataframe(
                            similarity_data,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Galerie des images similaires
                        st.markdown("#### Aperçu des correspondances")
                        cols = st.columns(min(len(top_files), 5))
                        
                        for idx, (file, score) in enumerate(zip(top_files, top_scores)):
                            if idx < 5:
                                with cols[idx]:
                                    try:
                                        # Construction du chemin
                                        possible_paths = [
                                            f"data/dataset_tache_1/dataset_tache_1/train/{file}",
                                            f"dataset/{file}",
                                            file
                                        ]
                                        
                                        img_path = None
                                        for path in possible_paths:
                                            if os.path.exists(path):
                                                img_path = path
                                                break
                                        
                                        if img_path:
                                            sim_img = Image.open(img_path)
                                            st.image(sim_img, use_column_width=True)
                                            st.caption(f"Similarité : {score*100:.1f}%")
                                        else:
                                            st.info(f"Fichier : {file}")
                                            st.caption(f"Similarité : {score*100:.1f}%")
                                    except:
                                        st.caption(f"Similarité : {score*100:.1f}%\nFichier : {file}")
                    else:
                        st.info("Aucun visage similaire trouvé dans la base de données.")
                else:
                    st.info("Base de données d'embeddings non disponible. Assurez-vous que le fichier embeddings_database.npy est présent dans le répertoire.")
                
                # Nettoyage du fichier temporaire
                os.unlink(temp_path)
                
            else:
                st.error("Impossible d'extraire les caractéristiques faciales. Veuillez vérifier que l'image contient un visage clairement visible.")
else:
    st.info("Veuillez télécharger une image contenant un visage pour commencer l'analyse")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
    Système de Reconnaissance Faciale | Technologies DeepFace & SVM
</div>
""", unsafe_allow_html=True)
