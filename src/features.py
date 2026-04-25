"""
Módulo de feature engineering — Hey Datathon 2026.

Contiene la lógica de gaps, etiquetado de churn y construcción de features RFM.
Todas las constantes están en la parte superior para que los notebooks
puedan importarlas y reutilizarlas sin redefinirlas.
"""
import pandas as pd
import numpy as np

# ── Constantes de etiquetado ──────────────────────────────────────────────────
# Sincronizadas con las definiciones del notebook 02_churn_labels.ipynb
GAP_RECOVERED_MIN   = 45   # días: inicio del rango de gap "recuperable"
GAP_RECOVERED_MAX   = 90   # días: fin del rango de gap "recuperable"
GAP_CHURNED         = 90   # días: a partir de aquí el cliente se considera perdido
VENTANA_ACTIVO_DIAS = 30   # días: ventana de actividad reciente para "healthy"

# Canales que se clasifican como uso de app móvil
APP_CANALES = {'ios', 'android', 'huawei', 'app_ios', 'app_android', 'app_huawei'}

# Variables de perfil del cliente que entran al modelo
FEATURES_PERFIL = [
    'score_buro',
    'satisfaccion_1_10',
    'dias_desde_ultimo_login',
    'num_productos_activos',
    'ingreso_mensual_mxn',
    'es_hey_pro',
    'nomina_domiciliada',
    'antiguedad_dias',
    'edad',
]

# Columnas que se excluyen de la matriz X (son target, ID o derivadas del etiquetado)
EXCLUIR_DE_X = {
    'user_id', 'churn', 'etiqueta',
    'fecha_ultima_tx', 'dias_desde_ultima_tx',
}


def calcular_gaps(df_transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el gap en días entre transacciones consecutivas de cada cliente.

    Parámetros
    ----------
    df_transacciones : DataFrame
        Debe tener columnas: user_id, fecha (o fecha_hora).

    Retorna
    -------
    DataFrame
        Copia ordenada por [user_id, fecha] con dos columnas adicionales:
        - 'fecha_anterior': timestamp de la TX previa del mismo cliente
        - 'gap_dias': días transcurridos desde la TX anterior (NaN en la primera TX)
    """
    # Detectar columna de fecha (normalizada por load_data, pero tolerante)
    date_col = 'fecha' if 'fecha' in df_transacciones.columns else 'fecha_hora'

    tx = df_transacciones.sort_values(['user_id', date_col]).copy()
    tx['fecha_anterior'] = tx.groupby('user_id')[date_col].shift(1)
    tx['gap_dias'] = (tx[date_col] - tx['fecha_anterior']).dt.days
    return tx


def _tiene_gap_y_regreso(grupo: pd.DataFrame) -> bool:
    """Devuelve True si el cliente tuvo un gap 45–90 días Y transaccionó después."""
    date_col = 'fecha' if 'fecha' in grupo.columns else 'fecha_hora'
    grupo = grupo.sort_values(date_col)
    for _, row in grupo.iterrows():
        if (pd.notna(row['gap_dias'])
                and GAP_RECOVERED_MIN <= row['gap_dias'] <= GAP_RECOVERED_MAX):
            if grupo[grupo[date_col] > row[date_col]].shape[0] > 0:
                return True
    return False


def etiquetar_clientes(df_transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna una etiqueta de churn a cada cliente según su historial de transacciones.

    Reglas (prioridad: recovered > churned > healthy)
    -------------------------------------------------
    'recovered' : tuvo un gap 45–90 días pero regresó a tener actividad posterior
    'churned'   : gap ≥ 90 días sin transacciones posteriores,
                  o lleva ≥ 90 días sin transaccionar desde la fecha de corte
    'healthy'   : sin gaps > 45 días y activo en los últimos 30 días del dataset

    Parámetros
    ----------
    df_transacciones : DataFrame
        Con columnas: user_id, fecha (datetime).

    Retorna
    -------
    DataFrame
        Una fila por cliente con columnas:
        user_id, etiqueta, max_gap_dias, fecha_ultima_tx,
        dias_desde_ultima_tx, es_recovered
    """
    date_col = 'fecha' if 'fecha' in df_transacciones.columns else 'fecha_hora'
    tx = calcular_gaps(df_transacciones)
    fecha_corte = tx[date_col].max()

    max_gap        = tx.groupby('user_id')['gap_dias'].max().rename('max_gap_dias')
    ultima_tx      = tx.groupby('user_id')[date_col].max().rename('fecha_ultima_tx')
    dias_inactivo  = ((fecha_corte - ultima_tx).dt.days).rename('dias_desde_ultima_tx')
    recovered_mask = tx.groupby('user_id').apply(_tiene_gap_y_regreso).rename('es_recovered')

    base = pd.DataFrame({
        'max_gap_dias':         max_gap,
        'fecha_ultima_tx':      ultima_tx,
        'dias_desde_ultima_tx': dias_inactivo,
        'es_recovered':         recovered_mask,
    }).reset_index()

    def _asignar(row):
        if row['es_recovered']:
            return 'recovered'
        if row['max_gap_dias'] >= GAP_CHURNED or row['dias_desde_ultima_tx'] >= GAP_CHURNED:
            return 'churned'
        if (row['dias_desde_ultima_tx'] <= VENTANA_ACTIVO_DIAS
                and row['max_gap_dias'] < GAP_RECOVERED_MIN):
            return 'healthy'
        return 'churned' if row['dias_desde_ultima_tx'] >= GAP_CHURNED else 'healthy'

    base['etiqueta'] = base.apply(_asignar, axis=1)
    return base


def build_rfm_features(df_transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 8 features RFM por cliente desde el historial de transacciones.

    Features calculadas
    -------------------
    recencia                       días desde última TX hasta fecha de corte
    frecuencia                     número total de transacciones
    monto_promedio                 promedio de monto por transacción (MXN)
    monto_total                    suma total de montos (MXN)
    max_gap_dias                   gap máximo entre transacciones consecutivas
    num_categorias_distintas       categorías MCC distintas utilizadas
    pct_transacciones_completadas  % de TXs con estatus 'completada'
    pct_uso_app                    % de TXs vía canal app (ios/android/huawei)

    Parámetros
    ----------
    df_transacciones : DataFrame
        Columnas obligatorias: user_id, fecha (datetime), monto.
        Opcionales: canal, estatus, categoria_mcc.

    Retorna
    -------
    DataFrame
        Una fila por user_id con las 8 features.
    """
    date_col = 'fecha' if 'fecha' in df_transacciones.columns else 'fecha_hora'
    tx = calcular_gaps(df_transacciones)
    fecha_corte = tx[date_col].max()

    # Canal app (múltiples naming conventions del CSV)
    if 'canal' in tx.columns:
        tx['es_app'] = tx['canal'].str.lower().isin(APP_CANALES).astype(int)
    else:
        tx['es_app'] = 0

    # Estatus completado
    if 'estatus' in tx.columns:
        tx['completada'] = (tx['estatus'].str.lower() == 'completada').astype(int)
    else:
        tx['completada'] = 1   # sin columna estatus → asumimos completadas

    rfm = tx.groupby('user_id').agg(
        recencia                     =(date_col,    lambda x: (fecha_corte - x.max()).days),
        frecuencia                   =(date_col,    'count'),
        monto_promedio               =('monto',     'mean'),
        monto_total                  =('monto',     'sum'),
        max_gap_dias                 =('gap_dias',  lambda x: x.max() if x.notna().any() else 0),
        pct_transacciones_completadas=('completada','mean'),
        pct_uso_app                  =('es_app',    'mean'),
    ).reset_index()

    # Diversidad de categorías MCC (opcional)
    if 'categoria_mcc' in df_transacciones.columns:
        n_cat = (df_transacciones.groupby('user_id')['categoria_mcc']
                 .nunique()
                 .rename('num_categorias_distintas')
                 .reset_index())
        rfm = rfm.merge(n_cat, on='user_id', how='left')
    else:
        rfm['num_categorias_distintas'] = 0

    rfm['max_gap_dias']             = rfm['max_gap_dias'].fillna(0)
    rfm['num_categorias_distintas'] = rfm['num_categorias_distintas'].fillna(0)
    return rfm


def build_model_features(
    df_clientes: pd.DataFrame,
    df_transacciones: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Combina features RFM con el perfil del cliente.

    Retorna el dataset listo para entrenar/evaluar el modelo de churn.

    Parámetros
    ----------
    df_clientes : DataFrame
        clientes_etiquetados.csv — debe incluir 'user_id', 'churn', 'etiqueta'
        y las columnas de FEATURES_PERFIL disponibles.
    df_transacciones : DataFrame
        hey_transacciones.csv (procesado por load_data).

    Retorna
    -------
    tuple[DataFrame, list[str]]
        df_model     : DataFrame con user_id + churn + etiqueta + todas las features
        feature_cols : columnas que entran como X al modelo (sin IDs/target)

    Imputaciones para clientes sin transacciones
    --------------------------------------------
    recencia=999, frecuencia=0, montos=0, gaps=0, porcentajes=0
    (Clientes que aparecen en clientes.csv pero no en transacciones)
    """
    rfm = build_rfm_features(df_transacciones)

    cols_base = ['user_id', 'churn', 'etiqueta'] + [
        c for c in FEATURES_PERFIL if c in df_clientes.columns
    ]
    df_model = df_clientes[cols_base].merge(rfm, on='user_id', how='left')

    _fill = {
        'recencia':                       999,
        'frecuencia':                     0,
        'monto_promedio':                 0,
        'monto_total':                    0,
        'max_gap_dias':                   0,
        'pct_transacciones_completadas':  0,
        'pct_uso_app':                    0,
        'num_categorias_distintas':       0,
    }
    for col, val in _fill.items():
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(val)

    feature_cols = [c for c in df_model.columns if c not in EXCLUIR_DE_X]
    return df_model, feature_cols
