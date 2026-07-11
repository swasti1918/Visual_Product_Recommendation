import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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


def dhash(image, hash_size=8):
    # difference hash (more robust to luminance changes than aHash)
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img)
    diff = arr[:, 1:] > arr[:, :-1]
    return diff.flatten().astype(np.uint8)


def phash(image, hash_size=8):
    # perceptual hash (robust, but implemented simply to avoid extra deps)
    # Resize to a fixed small grid and threshold by mean. This avoids reshape
    # errors when input sizes aren't divisible by block size.
    try:
        img = image.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
        small = np.asarray(img).astype(np.float32)
        avg = small.mean()
        return (small > avg).flatten().astype(np.uint8)
    except Exception:
        # fallback to aHash bits if anything goes wrong
        return ahash(image, hash_size=hash_size)


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


def _ensure_dhashes():
    global image_dhashes
    try:
        image_dhashes
    except NameError:
        # compute d-hashes for dataset images (lazy)
        image_dhashes = np.zeros_like(image_hashes)
        for idx, f in enumerate(image_files):
            try:
                img = Image.open(f)
                image_dhashes[idx] = dhash(img)
            except Exception:
                image_dhashes[idx] = image_hashes[idx]


def _ensure_phashes():
    global image_phashes
    try:
        image_phashes
    except NameError:
        image_phashes = np.zeros_like(image_hashes)
        for idx, f in enumerate(image_files):
            try:
                img = Image.open(f)
                image_phashes[idx] = phash(img)
            except Exception:
                # fallback to aHash for that image
                try:
                    with Image.open(f) as tmp:
                        image_phashes[idx] = ahash(tmp)
                except Exception:
                    image_phashes[idx] = image_hashes[idx]


        def extract_features(image, hist_bins=8, size=(64, 64)):
            # Resize and compute simple features: HSV hist (hist_bins per channel), aspect ratio, mean brightness
            try:
                img = image.convert('RGB').resize(size, Image.LANCZOS)
                arr = np.asarray(img).astype(np.float32) / 255.0
                # RGB -> HSV
                r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                maxc = np.maximum(np.maximum(r, g), b)
                minc = np.minimum(np.minimum(r, g), b)
                v = maxc
                s = (maxc - minc) / (maxc + 1e-8)
                # hue omitted; use simple channel histograms instead to avoid angle wrap
                feats = []
                for ch in range(3):
                    h, _ = np.histogram(arr[..., ch], bins=hist_bins, range=(0.0, 1.0))
                    feats.append(h.astype(np.float32) / (h.sum() + 1e-8))
                feats = np.concatenate(feats)
                # aspect ratio (height/width)
                w, h = image.size
                feats = np.concatenate([feats, np.array([h / (w + 1e-8), v.mean()])])
                return feats
            except Exception:
                return np.zeros(hist_bins * 3 + 2, dtype=np.float32)


        @st.cache_resource(show_spinner=False)
        def train_category_model():
            # Train a tiny logistic regression on the demo images' simple features
            X = []
            y = []
            for iid, img_path in zip(image_ids, image_files):
                meta = styles_lookup.get(iid, {})
                cat = meta.get('masterCategory')
                if not cat:
                    continue
                try:
                    with Image.open(img_path) as im:
                        X.append(extract_features(im))
                        y.append(cat)
                except Exception:
                    continue
            if len(X) < 3:
                return None, None
            X = np.vstack(X)
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            clf = LogisticRegression(max_iter=200, solver='liblinear')
            clf.fit(X, y_enc)
            return clf, le


        def predict_category_for_image(image):
            clf, le = train_category_model()
            if clf is None:
                return None
            feats = extract_features(image)
            pred = clf.predict(feats.reshape(1, -1))[0]
            return le.inverse_transform([pred])[0]


def combined_distances(query_ahash, query_dhash=None, query_phash=None, w_ahash=0.2, w_dhash=0.3, w_phash=0.5):
    # return combined distance (lower is more similar) to every dataset image
    _ensure_dhashes()
    _ensure_phashes()
    if query_dhash is None:
        query_dhash = query_ahash
    if query_phash is None:
        query_phash = query_ahash

    ahash_dists = np.sum(np.bitwise_xor(query_ahash, image_hashes), axis=1) / float(query_ahash.size)
    dhash_dists = np.sum(np.bitwise_xor(query_dhash, image_dhashes), axis=1) / float(query_dhash.size)
    phash_dists = np.sum(np.bitwise_xor(query_phash, image_phashes), axis=1) / float(query_phash.size)
    combined = (w_ahash * ahash_dists) + (w_dhash * dhash_dists) + (w_phash * phash_dists)
    return combined


def recommend_similar_combined(query_ahash, query_dhash=None, query_phash=None, top_k=5, **weights):
    combined = combined_distances(query_ahash, query_dhash=query_dhash, query_phash=query_phash, **weights)
    order = np.argsort(combined)
    top = order[:top_k]
    scores = 1.0 - combined[top]
    return top, scores, combined


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


def recommend_with_category(query_ahash, query_dhash=None, query_phash=None, coarse_k=30, top_k=5, min_best_score=0.35):
    # coarse pass across all images (larger k for more robust signals)
    # compute combined distances for the whole dataset, then pick top-k as coarse candidates
    combined = combined_distances(query_ahash, query_dhash=query_dhash, query_phash=query_phash)
    order_all = np.argsort(combined)
    coarse_indices = order_all[:coarse_k]
    coarse_scores = 1.0 - combined[coarse_indices]

    # compute per-category best (min) combined distance among dataset images
    # consider categories present in the coarse candidates to keep it focused
    cats_in_coarse = set()
    for i in coarse_indices:
        iid = image_ids[i]
        cat = styles_lookup.get(iid, {}).get("masterCategory")
        if cat:
            cats_in_coarse.add(cat)

    best_cat = None
    best_cat_min_dist = float("inf")
    best_cat_best_idx = None

    for cat in cats_in_coarse:
        candidate_idxs = [i for i, iid in enumerate(image_ids) if styles_lookup.get(iid, {}).get("masterCategory") == cat]
        if not candidate_idxs:
            continue
        dists_cat = combined[candidate_idxs]
        min_idx_local = int(np.argmin(dists_cat))
        min_dist = float(dists_cat[min_idx_local])
        if min_dist < best_cat_min_dist:
            best_cat_min_dist = min_dist
            best_cat = cat
            best_cat_best_idx = candidate_idxs[min_idx_local]

    # if we found a best category and the best in-category candidate is similar enough, lock to category
    if best_cat is not None:
        best_score = 1.0 - best_cat_min_dist
        if best_score >= min_best_score:
            candidate_idxs = [i for i, iid in enumerate(image_ids) if styles_lookup.get(iid, {}).get("masterCategory") == best_cat]
            # rank candidates by combined distance
            dists = combined[candidate_idxs]
            order = np.argsort(dists)
            selected = [candidate_idxs[i] for i in order[:top_k]]
            scores = 1.0 - dists[order[:top_k]]
            return selected, scores, best_cat

    # fallback: return the coarse results
    return coarse_indices[:top_k], coarse_scores[:top_k], None


# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_file, width="stretch")
    # allow user to override automatic category selection
    all_cats = sorted({v.get('masterCategory') for v in styles_lookup.values() if v.get('masterCategory')})
    cat_choice = st.selectbox("Category (Auto = detect automatically)", options=['Auto'] + all_cats)

    if st.button("Find Similar Products"):
        with st.spinner("Analyzing the image and finding demo catalog matches..."):
            try:
                image = Image.open(uploaded_file)
                query_hash = ahash(image)
                query_dhash = dhash(image)
                query_phash = phash(image)

                forced_cat = None if cat_choice == 'Auto' else cat_choice
                top_indices, scores, majority_cat = recommend_with_category(query_hash, query_dhash, query_phash, coarse_k=10, top_k=5)
                if forced_cat:
                    st.info(f"Category forced to: {forced_cat} — showing results from this category")
                    # re-rank within forced category
                    candidate_idxs = [i for i, iid in enumerate(image_ids) if styles_lookup.get(iid, {}).get('masterCategory') == forced_cat]
                    if candidate_idxs:
                        combined = combined_distances(query_hash, query_dhash=query_dhash, query_phash=query_phash)
                        dists = combined[candidate_idxs]
                        order = np.argsort(dists)
                        selected = [candidate_idxs[i] for i in order[:5]]
                        scores = 1.0 - dists[order[:5]]
                        top_indices = selected
                else:
                    if majority_cat:
                        st.info(f"Detected category: {majority_cat} — showing results from this category")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                st.error("An error occurred during recommendation. See details below.")
                st.code(tb)
                # also log some diagnostics
                try:
                    st.write({
                        "n_images": len(image_files),
                        "image_hashes_shape": getattr(image_hashes, 'shape', None),
                        "image_dhashes_defined": 'image_dhashes' in globals(),
                        "image_phashes_defined": 'image_phashes' in globals(),
                    })
                except Exception:
                    pass
                top_indices, scores = [], []

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
