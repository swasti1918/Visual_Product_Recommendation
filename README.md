# Visual Product Recommendation System

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge)](https://visualappuctrecommendation-jqitqky4yzkdghulenvxdy.streamlit.app)

## Overview

This project recommends visually similar fashion products using Deep Learning. It compares a baseline image embedding approach with a Siamese Network to improve recommendation quality.

## Features

- Image preprocessing
- Exploratory Data Analysis (EDA)
- Feature extraction using ResNet50
- Baseline recommendation system
- Siamese Network for improved embeddings
- Cosine similarity-based recommendations
- Streamlit web application

## Project Structure

```
Visual_Product_Recommender/
│── data/
│── embeddings/
│── models/
│── results/
│── Base Line recomm.ipynb
│── Dataset Preparation.ipynb
│── EDA.ipynb
│── Evaluate Baseline.ipynb
│── Generate image embedding.ipynb
│── Generate New Embeddings.ipynb
│── Improve Recommendation.ipynb
│── Siamese Network.ipynb
│── Transfer Learning.ipynb
│── Compare Results.ipynb
│── Streamlit App.ipynb
│── app.py
│── requirements.txt
│── README.md
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- Streamlit

## Dataset

This project is based on the **Fashion Product Images Dataset** from Kaggle.

**Dataset Link:**  
🔗 :contentReference[oaicite:0]{index=0}

> **Note:**  
> The deployed Streamlit application uses a **small dummy dataset** to reduce storage requirements and improve deployment speed. Therefore, the recommendations shown in the live demo may not be as accurate or diverse as those generated using the complete dataset.

To achieve the best recommendation quality:

1. Download the complete dataset from the Kaggle link above.
2. Replace the contents of the `data/` folder with the downloaded dataset.
3. Regenerate the image embeddings.
4. Restart the Streamlit application.

Using the full dataset significantly improves the quality and relevance of the recommended products. :contentReference[oaicite:1]{index=1}

## Results

- Implemented a baseline recommendation system using image embeddings.
- Improved recommendation quality using a Siamese Network.
- Generated visually similar fashion product recommendations.

## Future Improvements

- Deploy on Streamlit Cloud
- Use Vision Transformers (ViT)
- Add text-based search
- Hybrid recommendation system

## Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```
