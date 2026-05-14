Emotion-Aware Music Recommendation System

An intelligent music recommendation system that detects a user’s emotional state through facial expressions and recommends songs accordingly. This project explores two different deep learning pipelines for real-time emotion-aware music recommendation using computer vision, latent feature modeling, and natural language processing.

Overview

Traditional music recommendation systems mainly depend on listening history, ratings, or collaborative filtering techniques. This project focuses on incorporating real-time emotional context into the recommendation process by analyzing facial expressions captured through a webcam.

The project compares two independent pipelines:

Pipeline 1:  Vision Transformer + Variational Autoencoder
Facial emotion recognition using a pretrained Vision Transformer (ViT)
Music feature compression using a Variational Autoencoder (VAE)
Song retrieval using cosine similarity in latent space
Spotify audio feature dataset with 114,000 tracks

Pipeline 2 — ResNet18 + NLP-Based Recommendation
Facial emotion recognition using fine-tuned ResNet18
Multi-frame webcam emotion aggregation
Lyrics-based genre classification using TF-IDF and Multinomial Naive Bayes
Emotion-to-genre mapping for final song recommendation
Technologies Used
Python
PyTorch
TensorFlow / Keras
OpenCV
Hugging Face Transformers
Scikit-learn
NLTK
Pandas
NumPy
Datasets
FER2013 facial expression dataset
Spotify audio features dataset
Custom lyrics dataset across multiple music genres
Key Features
Real-time facial emotion detection
Emotion-aware song recommendation
Deep learning–based computer vision pipeline
NLP-based lyric and genre analysis
Comparative analysis of two recommendation architectures
Results

The project highlights both successful outcomes and practical challenges:

Pipeline 1 experienced posterior collapse in the VAE latent space.
Pipeline 2 produced semantically meaningful and consistent music recommendations.

Future Improvements

Larger lyrics corpus and transformer-based NLP models
User feedback integration for adaptive recommendations
Improved personalization and scalability
