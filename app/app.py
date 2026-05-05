import sys
import os

# 🔹 Fix import path FIRST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import torch

from src.preprocess import clean_text
from src.utils import build_pytorch_model, predict_pytorch

# =========================
# 🔹 Load shared resources
# =========================
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sklearn models
lr_model = joblib.load("models/lr_model.pkl")
nb_model = joblib.load("models/nb_model.pkl")

# PyTorch model
pytorch_model = build_pytorch_model()
pytorch_model.load_state_dict(torch.load("models/pytorch_model.pt"))
pytorch_model.eval()

# =========================
# 🔹 UI
# =========================
st.title("📩 Spam Detection App")
st.write("Compare multiple ML models on spam detection")

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Naive Bayes", "PyTorch Neural Network"]
)

user_input = st.text_area("Enter your message")

# =========================
# 🔹 Prediction logic
# =========================
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        if model_choice == "Logistic Regression":
            pred = lr_model.predict(vectorized)[0]
            prob = lr_model.predict_proba(vectorized)[0][1]

        elif model_choice == "Naive Bayes":
            pred = nb_model.predict(vectorized)[0]
            prob = nb_model.predict_proba(vectorized)[0][1]

        else:
            pred, prob = predict_pytorch(pytorch_model, vectorized)

        if pred == 1:
            st.error(f"🚨 Spam (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {1 - prob:.2f})")