from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


PROJECT_ROOT = Path(__file__).resolve().parents[3]

dataset_path = PROJECT_ROOT / "model" / "model_config" / "utils" / "dataset.csv"
model_path = PROJECT_ROOT / "model" / "model_config" / "utils" / "model.keras"
tokenizer_path = PROJECT_ROOT / "model" / "model_config" / "utils" / "tokenizer.pkl"
config_path = PROJECT_ROOT / "model" / "model_config" / "utils" / "config.pkl"


def get_dataset() -> pd.DataFrame:
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path, encoding='utf-8').dropna()

    else:
        from ..dataset_preparation.data_prep import create_dataset

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


early_stop = EarlyStopping(patience=5,
                           monitor='val_loss',
                           restore_best_weights=True)
model_check = ModelCheckpoint(save_best_only=True,
                              monitor='val_loss',
                              filepath=model_path)


model = models.Sequential([layers.Embedding(input_dim=max_words, output_dim=128),
                           layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
                           layers.Dropout(0.3),
                           layers.Dense(64, activation="relu"),
                           layers.Dropout(0.2),
                           layers.Dense(3, activation="softmax")])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train_pad, y_train,
                    validation_data=(X_val_pad, y_val),
                    epochs=24,
                    batch_size=256,
                    callbacks=[early_stop, model_check])


with open(config_path, 'wb') as f:
    pickle.dump({'max_len': max_len}, f)

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
