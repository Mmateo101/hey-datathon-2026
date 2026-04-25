"""
Módulo de carga y preprocesamiento de datos — Hey Datathon 2026.

Centraliza la lectura de los CSVs, el parseo de tipos, la normalización de
nombres de columna y los joins base. Todos los notebooks importan de aquí.
"""
import pandas as pd
from pathlib import Path

# Columnas que deben ser bool en hey_clientes.csv
_BOOL_COLS_CLIENTES = [
    'es_hey_pro', 'nomina_domiciliada', 'recibe_remesas',
    'usa_hey_shop', 'tiene_seguro', 'patron_uso_atipico',
]


def load_data(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los tres CSVs principales y aplica tipos correctos.

    Parámetros
    ----------
    base_path : str
        Carpeta que contiene los CSVs (e.g. 'data/raw/').

    Retorna
    -------
    tuple[DataFrame, DataFrame, DataFrame]
        (df_clientes, df_productos, df_transacciones)

        df_clientes:
            - Booleanos normalizados (es_hey_pro, nomina_domiciliada, …)
            - satisfaccion_1_10 puede ser NaN (751 clientes sin encuesta)
        df_productos:
            - Sin transformaciones; cargado tal cual
        df_transacciones:
            - Columna de fecha estandarizada como 'fecha' (datetime64)
              El CSV original puede usar el nombre 'fecha_hora' o 'fecha';
              ambos casos se normalizan a 'fecha'.
    """
    clientes = pd.read_csv(base_path + 'hey_clientes.csv')
    productos = pd.read_csv(base_path + 'hey_productos.csv')
    transacciones = pd.read_csv(base_path + 'hey_transacciones.csv')

    # ── Normalizar columna de fecha en transacciones ──────────────────────────
    # El CSV puede llamarla 'fecha_hora' o 'fecha'; estandarizamos a 'fecha'
    if 'fecha_hora' in transacciones.columns and 'fecha' not in transacciones.columns:
        transacciones = transacciones.rename(columns={'fecha_hora': 'fecha'})
    transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])

    # ── Asegurar booleanos correctos en clientes ──────────────────────────────
    for col in _BOOL_COLS_CLIENTES:
        if col in clientes.columns:
            clientes[col] = clientes[col].astype(bool)

    return clientes, productos, transacciones


def merge_clientes_etiquetados(
    df_clientes: pd.DataFrame,
    df_etiquetas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une el perfil de clientes con las etiquetas de churn.

    Parámetros
    ----------
    df_clientes : DataFrame
        Resultado de load_data()[0].
    df_etiquetas : DataFrame
        DataFrame con columnas: user_id, etiqueta, max_gap_dias,
        dias_desde_ultima_tx, fecha_ultima_tx.
        Producido por src.features.etiquetar_clientes().

    Retorna
    -------
    DataFrame
        Join left de clientes con etiquetas. Clientes sin transacciones
        (no presentes en df_etiquetas) reciben etiqueta = 'churned'.
    """
    cols_etiqueta = [
        c for c in ['user_id', 'etiqueta', 'max_gap_dias',
                    'dias_desde_ultima_tx', 'fecha_ultima_tx']
        if c in df_etiquetas.columns
    ]
    resultado = df_clientes.merge(df_etiquetas[cols_etiqueta], on='user_id', how='left')
    resultado['etiqueta'] = resultado['etiqueta'].fillna('churned')
    return resultado
