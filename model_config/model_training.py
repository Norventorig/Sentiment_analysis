import os
import pandas as pd


if os.path.exists(r"dataset"):
    dataset = pd.read_csv("dataset")

else:
    from dataset_preparation.data_prep import create_dataset

    create_dataset()
    dataset = pd.read_csv("dataset")
