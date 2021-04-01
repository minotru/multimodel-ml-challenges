from typing import Tuple

import pandas as pd

def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna("")
    
    return data

def select_popular_classes(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    label_column: str, 
    min_samples_per_class: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_train = data_train.groupby(label_column) \
        .filter(lambda g: g.size() >= min_samples_per_class)

    mask = data_test[label_column].isin(data_train[label_column])
    data_test = data_test[mask]

    return data_train, data_test

def select_sample(
   ,
    n_samples: int,
    min_samples_per_class: int)