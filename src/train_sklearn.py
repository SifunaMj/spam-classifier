import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

    print("✅ TF-IDF transformation complete")
    print("Shape of X:", X.shape)
    print("Sample features:", vectorizer.get_feature_names_out()[:20])