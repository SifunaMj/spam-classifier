# 🚀 Spam Detection System (ML + PyTorch)

## 📌 Project Overview

This project implements an end-to-end **SMS Spam Detection System** using both classical machine learning and deep learning approaches. The goal is to classify text messages as either:

* **Ham (0)** → Legitimate message
* **Spam (1)** → Unwanted/promotional message

The project is designed to demonstrate **real-world ML engineering practices**, including data preprocessing, feature engineering, model training, evaluation, and performance comparison.

---

## 🎯 Objectives

* Build a baseline using **TF-IDF + classical ML models**
* Implement a **PyTorch neural network**
* Compare models based on:

  * Accuracy
  * Precision / Recall / F1-score
  * Training time
* Derive meaningful insights from model performance

---

## 📂 Project Structure

```
spam-classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_sklearn.py
│   ├── train_pytorch.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│
├── app/ (optional Streamlit app)
│
├── README.md
└── requirements.txt
```

---

## 📥 Dataset

* **SMS Spam Collection Dataset**
* Contains labeled SMS messages:

  * `ham` (legitimate)
  * `spam` (unsolicited)

---

## 🔍 Exploratory Data Analysis

Key observations:

* Dataset is **imbalanced** (~86% ham, ~14% spam)
* Spam messages tend to:

  * Be longer
  * Contain promotional or urgent language

---

## 🧹 Data Preprocessing

Steps applied:

* Lowercasing text
* Removing punctuation
* Removing stopwords (via NLTK)
* Tokenization
* Cleaning empty/null entries

---

## 🔢 Feature Engineering

* Used **TF-IDF Vectorization**
* Limited to **3000 features**
* Converts text into numerical representation suitable for ML models

---

## 🧠 Models Implemented

### 1. 🔵 Naive Bayes

* Fast probabilistic model
* Assumes word independence
* Strong baseline for text classification

---

### 2. 🟢 Logistic Regression

* Linear model
* Widely used for classification tasks
* Serves as a robust baseline

---

### 3. 🔴 PyTorch Neural Network

Architecture:

* Input layer: 3000 features
* Hidden layers:

  * 128 units → ReLU → Dropout
  * 64 units → ReLU
* Output:

  * Sigmoid activation (binary classification)

Enhancements:

* Batch training
* Increased epochs (30)
* Dropout for regularization

---

## 📊 Final Results

| Model               | Accuracy | Spam Precision | Spam Recall | Spam F1  | Training Time |
| ------------------- | -------- | -------------- | ----------- | -------- | ------------- |
| Naive Bayes         | 0.9740   | **1.00**       | 0.81        | 0.89     | **0.002s**    |
| Logistic Regression | 0.9632   | 0.99           | 0.73        | 0.84     | 0.019s        |
| PyTorch NN          | 0.9740   | 0.91           | **0.89**    | **0.90** | 5.87s         |

---

## 🧠 Key Insights

### 1. Classical Models Are Strong Baselines

Naive Bayes performed exceptionally well despite its simplicity, achieving high accuracy with extremely low computation time.

---

### 2. Precision vs Recall Trade-off

* **Naive Bayes**:

  * Perfect precision (1.00)
  * Lower recall → misses some spam
* **PyTorch Model**:

  * Higher recall → detects more spam
  * Slightly lower precision

👉 This highlights a classic trade-off:

* Minimize **false positives** (Naive Bayes)
* Minimize **false negatives** (Neural Network)

---

### 3. Neural Networks Require Tuning

The PyTorch model initially underperformed but improved significantly after:

* Increasing epochs
* Introducing batch training
* Adding dropout and deeper layers

---

### 4. Performance vs Efficiency Trade-off

* **Naive Bayes**:

  * Fastest (milliseconds)
  * Suitable for real-time systems

* **PyTorch Model**:

  * Best balance (F1-score)
  * Computationally expensive (~6 seconds training)

---

### 5. Logistic Regression as Baseline

While stable and interpretable, Logistic Regression underperformed compared to the other models in this case.

---

## 🧾 Lessons Learned

* TF-IDF works extremely well with classical ML models
* Deep learning is not always superior for structured features
* Evaluation metrics (F1-score, recall) are more informative than accuracy in imbalanced datasets
* Model selection depends on **use case**, not just performance

---

## ⚖️ Model Selection Recommendations

| Use Case                 | Recommended Model      |
| ------------------------ | ---------------------- |
| Real-time / low-resource | Naive Bayes            |
| Balanced spam detection  | PyTorch Neural Network |
| Simple baseline          | Logistic Regression    |

---

## 🚀 Future Improvements

* Build a **Streamlit web app** for live predictions
* Experiment with:

  * Word embeddings (Word2Vec, GloVe)
  * Deep learning models (LSTM, Transformers)
* Handle class imbalance using:

  * Oversampling / undersampling
  * Class weights

---

## 🛠️ Technologies Used

* Python
* pandas, numpy
* scikit-learn
* PyTorch
* NLTK
* matplotlib / seaborn

---

## 🌍 Conclusion

This project demonstrates a complete machine learning workflow — from data preprocessing to model comparison — and highlights the importance of choosing the right model based on both **performance and practical constraints**.

---

## 🔗 Author

**Joseph Marvin Sifuna**

---

## ⭐ If you found this project useful, consider giving it a star!
