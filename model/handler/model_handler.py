from tensorflow.keras.preprocessing.sequence import pad_sequences


class ModelHandler:
    def __init__(self, model, tokenizer, pad_len, class_labels):
        self._model = model
        self._tokenizer = tokenizer
        self.pad_len = pad_len
        self.class_labels = class_labels

    def _prepare_data(self, x):
        x_seq = self._tokenizer.texts_to_sequences(x)
        x_pad_seq = pad_sequences(x_seq, maxlen=self.pad_len)

        return x_pad_seq

    def predict(self, x):
        try:
            x = self._prepare_data(x=x.astype(str))
            pred = self._model.predict(x).argmax(axis=1)
            pred = self.class_labels[pred]

        except:
            print('ВОЗНИКЛА ОШИБКА НА ЭТАПЕ ПРЕДСКАЗАНИЯ')

        else:
            return pred
