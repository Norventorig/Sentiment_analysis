import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
dataset_path = PROJECT_ROOT / "Sentiment_Analysis" / "model_config" / "dataset_preparation" / "datasets"


def create_dataset():
    df_1 = pd.read_csv(dataset_path / "data.csv")
    df_2 = pd.read_csv(dataset_path / 'sentiment_analysis.csv')
    df_3 = pd.read_csv(dataset_path / 'sentiment_data.csv')

    df_1['Sentiment'] = df_1['Sentiment'].apply(lambda x: {'positive': 2, 'neutral': 1, 'negative': 0}[x])
    df_1['Sentence'] = df_1['Sentence'].apply(lambda x: str.lower(x))
    df_1.rename(columns={'Sentence': 'text', 'Sentiment': 'sentiment'}, inplace=True)

    df_2 = df_2[['text', 'sentiment']]
    df_2['sentiment'] = df_2['sentiment'].apply(lambda x: {'positive': 2, 'neutral': 1, 'negative': 0}[x])
    df_2['text'] = df_2['text'].apply(lambda x: str.lower(x))

    df_3 = df_3[['Comment', 'Sentiment']]
    df_3.rename(columns={'Comment': 'text', 'Sentiment': 'sentiment'}, inplace=True)
    df_3.dropna(inplace=True)
    df_3['text'] = df_3['text'].apply(lambda x: str.lower(x))

    dataset = pd.concat([df_1, df_2, df_3])
    dataset.to_csv('dataset')


if __name__ == '__main__':
    create_dataset()
