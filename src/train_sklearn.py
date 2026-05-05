import pandas as pd
import time
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    return pd.read_csv(path)


def apply_tfidf(df):
    vectorizer = TfidfVectorizer(max_features=3000)

    # Clean data
    df = df.dropna(subset=['cleaned'])
    df = df[df['cleaned'].str.strip() != ""]

    X = vectorizer.fit_transform(df['cleaned'])

    # ✅ Consistent label encoding
    y = df['label'].map({'ham': 0, 'spam': 1})

    return X, y, vectorizer


if __name__ == "__main__":
    print("\n🚀 Starting Sklearn Training Pipeline")

    # 🔹 Load data
    df = load_data("data/processed/cleaned.csv")

    # 🔹 TF-IDF
    X, y, vectorizer = apply_tfidf(df)

    print("\n TF-IDF transformation complete")
    print("Shape of X:", X.shape)

    # 🔹 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n Train/Test Split:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("\n Label distribution (train):")
    print(y_train.value_counts(normalize=True))

    print("\n Label distribution (test):")
    print(y_test.value_counts(normalize=True))

    # =========================
    # 🔹 Naive Bayes
    # =========================
    print("\n🔵 Training Naive Bayes...")
    start = time.time()

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    nb_time = time.time() - start

    nb_preds = nb_model.predict(X_test)

    # =========================
    # 🔹 Logistic Regression
    # =========================
    print("\n🟢 Training Logistic Regression...")
    start = time.time()

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    lr_time = time.time() - start

    lr_preds = lr_model.predict(X_test)

    # =========================
    # 🔹 Save Models
    # =========================
    joblib.dump(nb_model, "models/nb_model.pkl")
    joblib.dump(lr_model, "models/lr_model.pkl")

    print("\n Models saved successfully")


    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("✅ Vectorizer saved successfully")

    # =========================
    # 🔹 Evaluation
    # =========================
    print("\n📊 Evaluation Results")

    # Naive Bayes
    print("\n🔵 Naive Bayes Performance:")
    print("Accuracy:", accuracy_score(y_test, nb_preds))
    print(classification_report(y_test, nb_preds))

    # Logistic Regression
    print("\n🟢 Logistic Regression Performance:")
    print("Accuracy:", accuracy_score(y_test, lr_preds))
    print(classification_report(y_test, lr_preds))

    # =========================
    # 🔹 Training Time
    # =========================
    print("\n⏱️ Training Time:")
    print(f"Naive Bayes: {nb_time:.4f} seconds")
    print(f"Logistic Regression: {lr_time:.4f} seconds")

    print("\n✅ Sklearn pipeline complete")