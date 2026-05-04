import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from train_sklearn import apply_tfidf, load_data


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    print(f"\n📌 {name} Evaluation")
    print("-" * 40)

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    # Load cleaned data
    df = load_data("data/processed/cleaned.csv")

    # Apply TF-IDF
    X, y, vectorizer = apply_tfidf(df)

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load models
    nb_model = joblib.load("models/nb_model.pkl")
    lr_model = joblib.load("models/lr_model.pkl")

    print("✅ Models loaded successfully")

    # Evaluate both models
    evaluate_model("Naive Bayes", nb_model, X_test, y_test)
    evaluate_model("Logistic Regression", lr_model, X_test, y_test)