"""
Utilidades de persistencia y scoring del modelo de churn — Hey Datathon 2026.
"""
import pickle
from pathlib import Path

# ── Umbrales de riesgo ────────────────────────────────────────────────────────
# Sincronizados con la celda 8 del notebook 03_modelo.ipynb.
# Modificar aquí actualiza automáticamente el comportamiento en notebooks.
UMBRAL_ALTO  = 0.70   # prob ≥ 0.70 → intervención inmediata
UMBRAL_MEDIO = 0.40   # prob ≥ 0.40 → nurturing preventivo
                      # prob <  0.40 → candidatos a cross-sell


def save_model(model, path: str) -> str:
    """
    Guarda el modelo entrenado como archivo .pkl.

    Parámetros
    ----------
    model : objeto sklearn/xgboost serializable con pickle
    path  : ruta completa de destino (e.g. 'models/churn_model.pkl')

    Retorna
    -------
    str : ruta donde se guardó el archivo (misma que path)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path


def load_model(path: str):
    """
    Carga un modelo guardado con save_model().

    Parámetros
    ----------
    path : str
        Ruta al archivo .pkl.

    Retorna
    -------
    Objeto modelo deserializado.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_risk_level(probability: float) -> str:
    """
    Clasifica la probabilidad de churn en un nivel de riesgo operativo.

    Niveles
    -------
    'Alto'  : prob >= 0.70  →  intervención inmediata (llamada, cashback, Hey Pro)
    'Medio' : prob >= 0.40  →  nurturing preventivo (push, recordatorio de beneficios)
    'Bajo'  : prob <  0.40  →  oportunidad de cross-sell

    Parámetros
    ----------
    probability : float
        Valor en [0, 1] devuelto por predict_proba.

    Retorna
    -------
    str : 'Alto', 'Medio' o 'Bajo'
    """
    if probability >= UMBRAL_ALTO:
        return 'Alto'
    elif probability >= UMBRAL_MEDIO:
        return 'Medio'
    else:
        return 'Bajo'
