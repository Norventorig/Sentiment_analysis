from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


class ModelHandler:
    def __init__(self, model, tokenizer, pad_len, class_labels):
        self._model = model
        self._tokenizer = tokenizer
        self.pad_len = pad_len
        self.class_labels = class_labels

    def _prepare_data(self, x):
        x = x.lower()
        x = re.sub(r'<.*?>', '', x)
        x = re.sub(r'[^a-z\s]', '', x)
        x = ' '.join([w for w in x.split() if w not in stop_words])

        x_seq = self._tokenizer.texts_to_sequences([x])
        x_pad_seq = pad_sequences(x_seq, maxlen=self.pad_len)

        return x_pad_seq

    def predict(self, x):
        try:
            if isinstance(x, str):
                x = self._prepare_data(x=x)
                prediction = self._model.predict(x).argmax(axis=1)[0]
                prediction = self.class_labels[prediction]

            else:
                raise ValueError('ТИП ВВЕДЕННЫХ ДАННЫХ ДОЛЖЕН БЫТЬ str')

        except Exception as e:
            print('ВОЗНИКЛА ОШИБКА НА ЭТАПЕ ПРЕДСКАЗАНИЯ', e)

        else:
            return prediction
