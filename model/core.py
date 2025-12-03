from pathlib import Path
from model.handler.model_handler import ModelHandler
from tensorflow.keras.models import load_model
import pickle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
utils_path = PROJECT_ROOT / "model" / "utils"

class_labels = ["negative", 'neutral', "positive"]
model = load_model(utils_path / "model.keras")

with open(utils_path / "tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
with open(utils_path / "config.pkl", 'rb') as f:
    pad_len = pickle.load(f)['max_len']


model_handler = ModelHandler(model=model, tokenizer=tokenizer, pad_len=pad_len, class_labels=class_labels)
