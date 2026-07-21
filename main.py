import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 Model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title("Visual Product Recommender System")


# Save Uploaded Image
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:
        st.error(e)
        return None



# Feature Extraction
def feature_extraction(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# Recommendation Function
def recommend(features, feature_list):

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm='brute',
        metric='cosine'
    )

    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# Upload Image
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:

    saved_path = save_uploaded_file(uploaded_file)

    if saved_path:

        display_image = Image.open(uploaded_file)

        st.image(display_image, caption="Uploaded Image", width=250)

        with st.spinner("Finding similar products..."):

            features = feature_extraction(saved_path, model)

            indices = recommend(features, feature_list)

        st.subheader("Recommended Products")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])

        with col2:
            st.image(filenames[indices[0][2]])

        with col3:
            st.image(filenames[indices[0][3]])

        with col4:
            st.image(filenames[indices[0][4]])

        with col5:
            st.image(filenames[indices[0][5]])

    else:
        st.error("Error while uploading image.")
