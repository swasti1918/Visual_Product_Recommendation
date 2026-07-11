import os
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

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
    "This public demo uses a lightweight image similarity model. "
    "Upload an image and see visually similar fashion products from the demo set."
)

# -----------------------------
# Paths and Data
# -----------------------------
IMAGE_PATH = "data/demo_images_public"
STYLES_PATH = "data/demo_subset_public.csv"


def image_feature_vector(image: Image.Image, bins=8):
    image = image.resize((224, 224)).convert("RGB")
    hist = np.array([])
    for channel in range(3):
        channel_data = np.array(image)[:, :, channel]
        hist_channel, _ = np.histogram(channel_data, bins=bins, range=(0, 256), density=True)
        hist = np.concatenate([hist, hist_channel])
    return hist


@st.cache_resource(show_spinner=False)
def load_demo_data():
    styles = pd.read_csv(STYLES_PATH)
    styles["id"] = styles["id"].astype(str)
    styles_lookup = styles.set_index("id").to_dict(orient="index")

    image_files = []
    image_vectors = []
    image_ids = []

    for image_id in styles["id"]:
        image_name = f"{image_id}.webp"
        image_path = os.path.join(IMAGE_PATH, image_name)
        if os.path.exists(image_path):
            image_files.append(image_path)
            image_ids.append(image_id)
            with Image.open(image_path) as img:
                image_vectors.append(image_feature_vector(img))

    image_vectors = np.vstack(image_vectors)
    return styles_lookup, image_ids, image_files, image_vectors


styles_lookup, image_ids, image_files, image_vectors = load_demo_data()


def recommend_similar(query_vector, top_k=5):
    similarity = cosine_similarity(query_vector.reshape(1, -1), image_vectors)[0]
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
            image = Image.open(uploaded_file)
            query_vector = image_feature_vector(image)
            top_indices, scores = recommend_similar(query_vector)

        st.subheader("Top Similar Products")
        cols = st.columns(min(5, len(top_indices)))

        for col, idx, score in zip(cols, top_indices, scores):
            image_id = image_ids[idx]
            product = styles_lookup.get(image_id, {})
            image_name = f"{image_id}.webp"
            image_path = os.path.join(IMAGE_PATH, image_name)
            title = product.get("productDisplayName", image_name)

            with col:
                st.image(image_path, width="stretch")
                st.caption(f"{title[:80]}{'...' if len(title) > 80 else ''}")
                st.caption(f"Similarity: {score:.3f}")
else:
    st.info("Upload an image to see fashion items from the public demo catalog.")
