import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def aggregate_transactions(transacciones: pd.DataFrame, id_col: str = "cliente_id") -> pd.DataFrame:
    return transacciones.groupby(id_col).agg(
        total_transacciones=("monto", "count"),
        monto_total=("monto", "sum"),
        monto_promedio=("monto", "mean"),
        monto_max=("monto", "max"),
    ).reset_index()


def scale_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df = df.copy()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler
