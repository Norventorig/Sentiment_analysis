from model.handler.model_handler import ModelHandler
from tensorflow.keras.models import load_model
import pickle


class_labels = ["negative", 'neutral', "positive"]
model = load_model(r"model_config\utils\model.keras")

with open(r"model_config\utils\tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
with open(r"model_config\utils\config.pkl", 'rb') as f:
    pad_len = pickle.load(f)['max_len']


model_handler = ModelHandler(model=model, tokenizer=tokenizer, pad_len=pad_len, class_labels=class_labels)
