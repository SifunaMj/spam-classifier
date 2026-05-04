import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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