from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
X = dataset["text"].astype(str)
y = dataset["sentiment"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

max_words = 20000
max_len = 20

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
