import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # 3. Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # 4. Tokenize (split into words)
    words = text.split()

    # 5. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 6. Join back into string
    return ' '.join(words)


def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path, encoding="ISO-8859-1")

    # Keep only first two columns (important for this dataset)
    df = df.iloc[:, :2]

    # Rename them
    df.columns = ['label', 'message']

    # Apply cleaning
    df['cleaned'] = df['message'].apply(clean_text)

    # Save processed data
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing complete. Saved to:", output_path)


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/spam.csv",
        output_path="data/processed/cleaned.csv"
    )