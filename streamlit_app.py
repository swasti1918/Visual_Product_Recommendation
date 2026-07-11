import os
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Visual Product Recommendation Demo",
    page_icon="👕",
    layout="wide",
)

st.title("👕 Visual Product Recommendation Demo")
st.write(
    "This demo uses ResNet50 image embeddings for more accurate semantic matches."
)

# -----------------------------
# Paths and Data
# -----------------------------
IMAGE_PATH = "data/demo_images_public"
EMBEDDINGS_PATH = "embeddings/demo_image_embeddings_public.npy"
IMAGE_IDS_PATH = "embeddings/demo_image_ids_public.npy"
STYLES_PATH = "data/demo_subset_public.csv"


@st.cache_resource(show_spinner=False)
def load_recommender():
    feature_extractor = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )

    embeddings = np.load(EMBEDDINGS_PATH)
    image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=True)

    styles = pd.read_csv(STYLES_PATH)
    styles["id"] = styles["id"].astype(str)
    styles_lookup = styles.set_index("id").to_dict(orient="index")

    return feature_extractor, embeddings, image_ids, styles_lookup


feature_extractor, embeddings, image_ids, styles_lookup = load_recommender()


# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


def recommend_similar(query_embedding, top_k=5):
    similarity = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarity)[::-1][:top_k]
    return top_indices, similarity[top_indices]


# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_file, width="stretch")

    if st.button("Find Similar Products"):
        with st.spinner("Analyzing the image and finding demo catalog matches..."):
            query_embedding = feature_extractor.predict(
                preprocess_image(uploaded_file),
                verbose=0,
            )
            top_indices, scores = recommend_similar(query_embedding)

        st.subheader("Top Similar Products")
        cols = st.columns(min(5, len(top_indices)))

        for col, idx, score in zip(cols, top_indices, scores):
            image_id = int(image_ids[idx])
            image_name = f"{image_id}.webp"
            image_path = os.path.join(IMAGE_PATH, image_name)
            product = styles_lookup.get(str(image_id), {})
            title = product.get("productDisplayName", image_name)

            with col:
                st.image(image_path, width="stretch")
                st.caption(f"{title[:80]}{'...' if len(title) > 80 else ''}")
                st.caption(f"Similarity: {score:.3f}")
else:
    st.info("Upload an image to see fashion items from the public demo catalog.")
