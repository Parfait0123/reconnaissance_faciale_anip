import streamlit as st
import numpy as np
import joblib
from PIL import Image
import io
import os
import tempfile
import gdown
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import json
import glob

# Configuration de la page
st.set_page_config(
    page_title="Système de Reconnaissance Faciale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .similarity-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: #f1f3f4;
        border-radius: 8px;
        border-left: 3px solid #2196F3;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour télécharger les fichiers depuis Google Drive
@st.cache_resource
def download_from_drive(file_id, output_path):
    """Télécharge un fichier depuis Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Erreur lors du téléchargement: {str(e)}")
        return False

# Fonction pour télécharger et extraire le dataset
def download_and_extract_dataset(dataset_id, extract_path):
    """Télécharge et extrait le dataset"""
    try:
        with st.spinner("Téléchargement du dataset en cours..."):
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "dataset.zip")
            
            if not download_from_drive(dataset_id, zip_path):
                return False
            
            # Extraire le zip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            st.success("Dataset téléchargé et extrait avec succès")
            return True
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du dataset: {str(e)}")
        return False

# Fonction pour extraire les embeddings des images de test
def extract_test_embeddings(test_dir, embedding_model='ArcFace', detector='retinaface'):
    """Extrait les embeddings de toutes les images de test"""
    try:
        with st.spinner("Extraction des embeddings des images de test..."):
            all_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
            embeddings_dict = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_path in enumerate(all_files):
                try:
                    embedding_obj = DeepFace.represent(
                        img_path,
                        model_name=embedding_model,
                        enforce_detection=True,
                        detector_backend=detector
                    )
                    filename = os.path.basename(img_path)
                    embeddings_dict[filename] = np.array(embedding_obj[0]['embedding'])
                    
                    # Mettre à jour la progression
                    progress = (i + 1) / len(all_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Traitement de l'image {i+1}/{len(all_files)}: {filename}")
                    
                except Exception as e:
                    st.warning(f"Erreur d'extraction pour l'image {img_path}: {e}")
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"Extraction terminée: {len(embeddings_dict)} embeddings extraits")
            return embeddings_dict
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des embeddings: {str(e)}")
        return {}

# Fonction pour charger les modèles
@st.cache_resource
def load_models(classifier_id, encoder_id, embeddings_id):
    """Charge les modèles et données nécessaires"""
    try:
        with st.spinner("Chargement des modèles en cours..."):
            # Créer un répertoire temporaire
            temp_dir = tempfile.mkdtemp()
            
            # Chemins des fichiers
            classifier_path = os.path.join(temp_dir, "face_classifier.pkl")
            encoder_path = os.path.join(temp_dir, "label_encoder.pkl")
            embeddings_path = os.path.join(temp_dir, "embeddings_database.npy")
            
            # Télécharger les fichiers
            if not download_from_drive(classifier_id, classifier_path):
                return None, None, None
            if not download_from_drive(encoder_id, encoder_path):
                return None, None, None
            if embeddings_id and not download_from_drive(embeddings_id, embeddings_path):
                st.warning("Fichier d'embeddings non chargé. Le clustering sera désactivé.")
                embeddings_data = None
            else:
                # Charger les embeddings depuis le fichier .npy
                embeddings_array = np.load(embeddings_path, allow_pickle=True)
                # Convertir en format dictionnaire pour la compatibilité
                embeddings_data = {str(i): emb for i, emb in enumerate(embeddings_array)}
            
            # Charger les modèles
            classifier = joblib.load(classifier_path)
            label_encoder = joblib.load(encoder_path)
            
            st.success("Modèles chargés avec succès")
            return classifier, label_encoder, embeddings_data
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None, None

# Fonction pour extraire l'embedding d'une image
def extract_embedding(image, embedding_model='ArcFace', detector='retinaface'):
    """Extrait l'embedding facial d'une image"""
    try:
        # Sauvegarder temporairement l'image
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_image.jpg")
        image.save(temp_path)
        
        # Extraire l'embedding
        embedding_obj = DeepFace.represent(
            temp_path,
            model_name=embedding_model,
            enforce_detection=True,
            detector_backend=detector
        )
        
        # Nettoyer
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return np.array(embedding_obj[0]['embedding'])
    except Exception as e:
        raise Exception(f"Erreur lors de l'extraction: {str(e)}")

# Fonction pour trouver les visages similaires
def find_similar_faces(query_embedding, embeddings_data, top_k=5):
    """Trouve les visages les plus similaires via clustering"""
    if not embeddings_data:
        return []
    
    try:
        # Convertir les embeddings stockés
        stored_embeddings = []
        image_ids = []
        
        for img_id, embedding in embeddings_data.items():
            stored_embeddings.append(embedding)
            image_ids.append(img_id)
        
        stored_embeddings = np.array(stored_embeddings)
        
        # Calculer les similarités cosinus
        similarities = cosine_similarity([query_embedding], stored_embeddings)[0]
        
        # Obtenir les top K plus similaires
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_id': image_ids[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    except Exception as e:
        st.error(f"Erreur lors de la recherche de similarités: {str(e)}")
        return []

# Interface principale
def main():
    # En-tête
    st.markdown('<p class="main-header">Système de Reconnaissance Faciale</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse avancée par classification et clustering</p>', unsafe_allow_html=True)
    
    # Barre latérale pour la configuration
    with st.sidebar:
        st.header("Configuration des Modèles")
        
        st.markdown("""
        <div class="info-box">
        <strong>Instructions:</strong><br>
        Entrez les identifiants Google Drive des fichiers de modèles ci-dessous.
        </div>
        """, unsafe_allow_html=True)
        
        # Section pour le dataset
        st.subheader("Dataset")
        dataset_id = st.text_input(
            "ID Google Drive - Dataset (optionnel)",
            placeholder="1ABC...XYZ",
            help="ID du fichier dataset.zip"
        )
        
        if dataset_id and st.button("Télécharger le Dataset"):
            temp_data_dir = tempfile.mkdtemp()
            if download_and_extract_dataset(dataset_id, temp_data_dir):
                st.session_state.dataset_path = temp_data_dir
                # Extraire automatiquement les embeddings après téléchargement
                test_dir = os.path.join(temp_data_dir, "data/dataset_tache_1/dataset_tache_1/test")
                if os.path.exists(test_dir):
                    embeddings_data = extract_test_embeddings(test_dir)
                    if embeddings_data:
                        st.session_state.embeddings_data = embeddings_data
                        st.success("Embeddings des images de test extraits avec succès")
        
        st.markdown("---")
        
        # Section pour les modèles
        st.subheader("Modèles")
        classifier_id = st.text_input(
            "ID Google Drive - Classificateur",
            placeholder="1ABC...XYZ",
            help="ID du fichier face_classifier.pkl"
        )
        
        encoder_id = st.text_input(
            "ID Google Drive - Encodeur",
            placeholder="1ABC...XYZ",
            help="ID du fichier label_encoder.pkl"
        )
        
        embeddings_id = st.text_input(
            "ID Google Drive - Embeddings (optionnel)",
            placeholder="1ABC...XYZ",
            help="ID du fichier embeddings_database.npy"
        )
        
        st.markdown("---")
        
        # Paramètres avancés
        with st.expander("Paramètres Avancés"):
            embedding_model = st.selectbox(
                "Modèle d'embedding",
                ["ArcFace", "Facenet", "VGG-Face"],
                index=0
            )
            
            detector = st.selectbox(
                "Détecteur de visage",
                ["retinaface", "mtcnn", "opencv"],
                index=0
            )
            
            top_k = st.slider(
                "Nombre de visages similaires",
                min_value=1,
                max_value=10,
                value=5
            )
        
        load_button = st.button("Charger les Modèles", type="primary")
        
        # Informations sur l'application
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.85rem; color: #666;">
        <strong>À propos</strong><br>
        Version 1.0<br>
        Développé pour l'analyse de reconnaissance faciale<br>
        Utilise DeepFace et SVM
        </div>
        """, unsafe_allow_html=True)
    
    # Charger les modèles si demandé
    if load_button:
        if not classifier_id or not encoder_id:
            st.error("Veuillez fournir au minimum les IDs du classificateur et de l'encodeur.")
        else:
            st.session_state.classifier, st.session_state.label_encoder, st.session_state.embeddings_data = load_models(
                classifier_id, encoder_id, embeddings_id
            )
            st.session_state.embedding_model = embedding_model
            st.session_state.detector = detector
            st.session_state.top_k = top_k
    
    # Vérifier si les modèles sont chargés
    if 'classifier' not in st.session_state or st.session_state.classifier is None:
        st.markdown("""
        <div class="warning-box">
        <strong>Aucun modèle chargé</strong><br>
        Veuillez configurer et charger les modèles dans la barre latérale pour commencer.
        </div>
        """, unsafe_allow_html=True)
        
        # Section d'information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fonctionnalités
            
            **Classification SVM**
            - Identification précise des personnes
            - Score de confiance pour chaque prédiction
            - Basé sur des embeddings ArcFace
            
            **Clustering par Similarité**
            - Recherche des visages les plus proches
            - Analyse par distance cosinus
            - Jusqu'à 10 résultats similaires
            """)
        
        with col2:
            st.markdown("""
            ### Comment utiliser
            
            1. Entrez les identifiants Google Drive dans la barre latérale
            2. Cliquez sur "Charger les Modèles"
            3. Téléchargez une image de test
            4. Analysez les résultats de classification et de clustering
            
            Les modèles doivent être au format `.pkl` pour le classificateur et l'encodeur,
            et `.npy` pour la base de données d'embeddings.
            """)
        
        return
    
    # Interface principale d'analyse
    st.markdown("---")
    
    # Upload d'image
    uploaded_file = st.file_uploader(
        "Télécharger une image à analyser",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image uploadée
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image téléchargée", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="result-box">
            <h3>Informations sur l'image</h3>
            """, unsafe_allow_html=True)
            
            st.write(f"**Nom du fichier:** {uploaded_file.name}")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'N/A'}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Bouton d'analyse
        if st.button("Analyser l'image", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Extraire l'embedding
                    embedding = extract_embedding(
                        image,
                        st.session_state.embedding_model,
                        st.session_state.detector
                    )
                    
                    # Prédiction avec le classificateur
                    prediction_encoded = st.session_state.classifier.predict([embedding])
                    predicted_person = st.session_state.label_encoder.inverse_transform(prediction_encoded)[0]
                    
                    # Probabilités
                    probabilities = st.session_state.classifier.predict_proba([embedding])
                    max_prob = np.max(probabilities) * 100
                    
                    # Afficher les résultats de classification
                    st.markdown("---")
                    st.markdown("## Résultats de l'Analyse")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h4>Identité Prédite</h4>
                        <h2 style="color: #4CAF50; margin: 0;">{predicted_person}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h4>Confiance</h4>
                        <h2 style="color: #2196F3; margin: 0;">{max_prob:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_level = "Élevée" if max_prob > 70 else "Moyenne" if max_prob > 50 else "Faible"
                        color = "#4CAF50" if max_prob > 70 else "#ff9800" if max_prob > 50 else "#f44336"
                        st.markdown(f"""
                        <div class="metric-card">
                        <h4>Niveau de Confiance</h4>
                        <h2 style="color: {color}; margin: 0;">{confidence_level}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top 5 prédictions
                    st.markdown("### Distribution des Probabilités")
                    top_5_indices = np.argsort(probabilities[0])[::-1][:5]
                    
                    for idx in top_5_indices:
                        person = st.session_state.label_encoder.inverse_transform([idx])[0]
                        prob = probabilities[0][idx] * 100
                        
                        st.markdown(f"""
                        <div class="similarity-item">
                        <strong>{person}</strong>
                        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; margin-top: 5px;">
                            <div style="background-color: #4CAF50; width: {prob}%; height: 100%; border-radius: 10px; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
                                {prob:.2f}%
                            </div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Analyse par clustering
                    if 'embeddings_data' in st.session_state and st.session_state.embeddings_data:
                        st.markdown("---")
                        st.markdown("### Visages Similaires (Clustering)")
                        
                        similar_faces = find_similar_faces(
                            embedding,
                            st.session_state.embeddings_data,
                            st.session_state.top_k
                        )
                        
                        if similar_faces:
                            st.markdown(f"**{len(similar_faces)} visage(s) similaire(s) trouvé(s)**")
                            
                            for i, face in enumerate(similar_faces, 1):
                                similarity_score = face['similarity'] * 100
                                st.markdown(f"""
                                <div class="similarity-item">
                                <strong>#{i} - Image ID: {face['image_id']}</strong><br>
                                Similarité: <strong style="color: #2196F3;">{similarity_score:.2f}%</strong>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Aucun visage similaire trouvé dans la base de données.")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")
                    st.markdown("""
                    <div class="warning-box">
                    <strong>Suggestions:</strong><br>
                    - Vérifiez que l'image contient un visage clairement visible<br>
                    - Assurez-vous que l'image est de bonne qualité<br>
                    - Réessayez avec une autre image
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
