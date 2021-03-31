import pandas as pd

def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna("")
    
    return data