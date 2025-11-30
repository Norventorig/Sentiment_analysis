import os
import pandas as pd


def get_dataset() -> pd.DataFrame:
    if os.path.exists(r"dataset"):
        data = pd.read_csv("dataset")

    else:
        from dataset_preparation.data_prep import create_dataset

        create_dataset()
        data = pd.read_csv("dataset")

    return data


dataset = get_dataset()
