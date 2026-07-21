import os
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm

# -------------------------------
# Load ResNet50 Model
# -------------------------------
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# Image Folder
# -------------------------------
IMAGE_FOLDER = "images_small"

filenames = []
feature_list = []

# Get all image paths
for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        filenames.append(os.path.join(IMAGE_FOLDER, file))

print(f"Found {len(filenames)} images.")

# -------------------------------
# Extract Features
# -------------------------------
for i, file in enumerate(filenames):

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()

    normalized_result = result / norm(result)

    feature_list.append(normalized_result)

    print(f"{i+1}/{len(filenames)} Processed")

# -------------------------------
# Save Embeddings
# -------------------------------
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("\n✅ Embeddings Generated Successfully!")
print(f"Total Images: {len(filenames)}")
