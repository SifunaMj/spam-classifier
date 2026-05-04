import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib

from sklearn.metrics import accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    return df

def apply_tfidf(df):
    vectorizer = TfidfVectorizer(max_features=3000)

    # Clean data before vectorizing
    df = df.dropna(subset=['cleaned'])
    df = df[df['cleaned'].str.strip() != ""]

    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    return X, y, vectorizer



if __name__ == "__main__":
    # Load cleaned data
    df = load_data("data/processed/cleaned.csv")

    # Apply TF-IDF
    X, y, vectorizer = apply_tfidf(df)

    # 🔹 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("✅ TF-IDF transformation complete")
    print("Shape of X:", X.shape)

    print("\n📊 Train/Test Split:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("\n📊 Label distribution (train):")
    print(y_train.value_counts(normalize=True))

    print("\n📊 Label distribution (test):")
    print(y_test.value_counts(normalize=True))

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Predict
    nb_preds = nb_model.predict(X_test)

    print("\n✅ Naive Bayes trained")


    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Predict
    lr_preds = lr_model.predict(X_test)

    print("✅ Logistic Regression trained")

    joblib.dump(nb_model, "models/nb_model.pkl")
    joblib.dump(lr_model, "models/lr_model.pkl")

    print("\n💾 Models saved successfully")
    print("\n📊 Quick Accuracy Check:")
    print("Naive Bayes:", accuracy_score(y_test, nb_preds))
    print("Logistic Regression:", accuracy_score(y_test, lr_preds))

    