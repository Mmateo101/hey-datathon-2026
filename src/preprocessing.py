import pandas as pd
import numpy as np


def load_datasets(data_path: str) -> dict[str, pd.DataFrame]:
    return {
        "clientes": pd.read_csv(data_path + "hey_clientes.csv"),
        "productos": pd.read_csv(data_path + "hey_productos.csv"),
        "transacciones": pd.read_csv(data_path + "hey_transacciones.csv"),
        "anonimizado": pd.read_parquet(data_path + "dataset_50k_anonymized.parquet"),
    }


def drop_duplicates_and_nulls(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = df.drop_duplicates()
    null_ratio = df.isnull().mean()
    cols_to_keep = null_ratio[null_ratio < threshold].index
    return df[cols_to_keep]
