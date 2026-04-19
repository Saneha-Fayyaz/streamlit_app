import streamlit as st
import numpy as np
import cv2
import pickle
import os
import base64
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocess import preprocess_bytes
from scripts.feature_extractor import FeatureExtractor

FEATURES_FILE = 'saved_features/features.pkl'

# ── Globals ────────────────────────────────────────────────────────────────
_extractor = None
_catalog_paths = None
_catalog_features = None
_index_loaded = False


def get_extractor():
    global _extractor
    if _extractor is None:
        print("[App] Loading feature extractor...")
        _extractor = FeatureExtractor()
    return _extractor


def load_index():
    global _catalog_paths, _catalog_features, _index_loaded
    if _index_loaded:
        return _catalog_paths, _catalog_features
    if not os.path.exists(FEATURES_FILE):
        return None, None
    try:
        with open(FEATURES_FILE, 'rb') as f:
            data = pickle.load(f)
        _catalog_paths = data['paths']
        _catalog_features = data['features']
        _index_loaded = True
        print(f"[App] Index loaded: {len(_catalog_paths)} rings")
        return _catalog_paths, _catalog_features
    except Exception as e:
        print(f"[App] Index load error: {e}")
        return None, None


st.set_page_config(page_title="LUMIÈRE Visual Search", layout="centered")

st.title("💎 LUMIÈRE Visual Ring Search")
st.write("Upload a ring image to find similar designs")

# Load features
@st.cache_resource
def load_features():
    with open("saved_features/features.pkl", "rb") as f:
        return pickle.load(f)

features = load_features()

# Upload image
uploaded_file = st.file_uploader("Upload Ring Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocess the image
    img_bytes = uploaded_file.getvalue()
    query_img_data = preprocess_bytes(img_bytes)

    if query_img_data is None:
        st.error("Image preprocessing failed.")
    else:
        # Load index
        catalog_paths, catalog_features = load_index()
        if catalog_paths is None:
            st.error("Feature index not found. Please ensure features.pkl exists.")
        else:
            extractor = get_extractor()
            if extractor.model is None:
                st.error("ML model failed to load.")
            else:
                # Extract features
                query_features = extractor.extract_with_augmentation(query_img_data, n_aug=4)
                if query_features is None:
                    query_features = extractor.extract(query_img_data)
                if query_features is None:
                    st.error("Feature extraction failed.")
                else:
                    # Compute similarity
                    cos_sims = cosine_similarity(
                        query_features.reshape(1, -1),
                        catalog_features
                    )[0]

                    top_n = min(6, len(catalog_paths))
                    top_indices = np.argsort(cos_sims)[-top_n:][::-1]

                    st.write("🔍 Finding similar rings...")

                    # Display results
                    st.success("Top matches:")

                    for i, idx in enumerate(top_indices, 1):
                        path = catalog_paths[idx]
                        filename = os.path.basename(path)
                        similarity = float(cos_sims[idx])

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if os.path.exists(path):
                                result_image = Image.open(path)
                                st.image(result_image, caption=f"Match {i}", use_column_width=True)
                            else:
                                st.write(f"Match {i}: Image not found")

                        with col2:
                            st.write(f"**Similarity:** {similarity * 100:.1f}%")
                            st.write(f"**File:** {filename}")

