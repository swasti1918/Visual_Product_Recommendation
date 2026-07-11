# Visual Product Recommendation System

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

Fashion Product Images Dataset (Kaggle)

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