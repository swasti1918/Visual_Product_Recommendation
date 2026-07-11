import os
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps
import numpy as np

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
STYLES_PATH = "data/demo_subset_public.csv"


def ahash(image: Image.Image, hash_size=8):
    """Compute average hash (aHash) for an image as a boolean array."""
    image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(image).astype(np.float32)
    avg = pixels.mean()
    return (pixels > avg).flatten()


@st.cache_resource(show_spinner=False)
def load_demo_hashes():
    styles = pd.read_csv(STYLES_PATH)
    styles["id"] = styles["id"].astype(str)
    styles_lookup = styles.set_index("id").to_dict(orient="index")

    image_ids = []
    image_files = []
    hashes = []

    for image_id in styles["id"]:
        image_name = f"{image_id}.webp"
        image_path = os.path.join(IMAGE_PATH, image_name)
        if os.path.exists(image_path):
            image_ids.append(image_id)
            image_files.append(image_path)
            with Image.open(image_path) as img:
                hashes.append(ahash(img))

    hashes = np.vstack(hashes)
    return styles_lookup, image_ids, image_files, hashes


styles_lookup, image_ids, image_files, image_hashes = load_demo_hashes()


def recommend_similar_hash(query_hash, top_k=5):
    # Hamming distance
    dists = np.sum(np.bitwise_xor(query_hash, image_hashes), axis=1)
    top_indices = np.argsort(dists)[:top_k]
    scores = 1.0 - (dists[top_indices] / (query_hash.size))
    return top_indices, scores


def get_majority_category(indices):
    from collections import Counter
    cats = []
    for i in indices:
        iid = image_ids[i]
        meta = styles_lookup.get(iid, {})
        cat = meta.get("masterCategory")
        if cat:
            cats.append(cat)
    if not cats:
        return None, 0
    most, cnt = Counter(cats).most_common(1)[0]
    return most, cnt


def recommend_with_category(query_hash, coarse_k=10, top_k=5):
    # coarse pass across all images
    coarse_indices, coarse_scores = recommend_similar_hash(query_hash, top_k=coarse_k)
    majority_cat, cnt = get_majority_category(coarse_indices)

    # require a strict majority to trust category (more than half)
    if majority_cat is not None and cnt >= (coarse_k // 2) + 1:
        # filter candidates to same masterCategory
        candidate_idxs = [i for i, iid in enumerate(image_ids) if styles_lookup.get(iid, {}).get("masterCategory") == majority_cat]
        if len(candidate_idxs) >= 1:
            # compute distances only within candidates
            dists = np.sum(np.bitwise_xor(query_hash, image_hashes[candidate_idxs]), axis=1)
            order = np.argsort(dists)
            selected = [candidate_idxs[i] for i in order[:top_k]]
            scores = 1.0 - (dists[order[:top_k]] / (query_hash.size))
            return selected, scores, majority_cat

    # fallback: return the coarse results
    return coarse_indices[:top_k], coarse_scores[:top_k], None


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
            query_hash = ahash(image)
            top_indices, scores, majority_cat = recommend_with_category(query_hash, coarse_k=10, top_k=5)
            if majority_cat:
                st.info(f"Detected category: {majority_cat} — showing results from this category")

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
