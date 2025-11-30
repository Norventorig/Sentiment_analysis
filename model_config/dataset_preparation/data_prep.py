import pandas as pd
from pathlib import Path
import re
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
dataset_path = PROJECT_ROOT / "Sentiment_Analysis" / "model_config" / "dataset_preparation" / "datasets"

stop_words = set(stopwords.words('english'))


def create_dataset():
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([w for w in text.split() if w not in stop_words])
        return text

    df_1 = pd.read_csv(dataset_path / "data.csv")
    df_2 = pd.read_csv(dataset_path / 'sentiment_analysis.csv')
    df_3 = pd.read_csv(dataset_path / 'sentiment_data.csv')

    df_1['Sentiment'] = df_1['Sentiment'].apply(lambda x: {'positive': 2, 'neutral': 1, 'negative': 0}[x])
    df_1['Sentence'] = df_1['Sentence'].apply(clean_text)
    df_1.rename(columns={'Sentence': 'text', 'Sentiment': 'sentiment'}, inplace=True)

    df_2 = df_2[['text', 'sentiment']]
    df_2['sentiment'] = df_2['sentiment'].apply(lambda x: {'positive': 2, 'neutral': 1, 'negative': 0}[x])
    df_2['text'] = df_2['text'].apply(clean_text)

    df_3 = df_3[['Comment', 'Sentiment']]
    df_3.rename(columns={'Comment': 'text', 'Sentiment': 'sentiment'}, inplace=True)
    df_3.dropna(inplace=True)
    df_3['text'] = df_3['text'].apply(clean_text)

    dataset = pd.concat([df_1, df_2, df_3])
    dataset.to_csv('dataset')


if __name__ == '__main__':
    create_dataset()
