from pathlib import Path
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dataset_path = PROJECT_ROOT / "Sentiment_Analysis" / "model_config" / "dataset.csv"


def get_dataset() -> pd.DataFrame:
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path, encoding='utf-8').dropna()

    else:
        from dataset_preparation.data_prep import create_dataset

        create_dataset()
        data = pd.read_csv(dataset_path, encoding='utf-8').dropna()

    return data


dataset = get_dataset()

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['sentiment']
