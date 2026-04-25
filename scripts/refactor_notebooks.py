"""
Actualiza las celdas modificadas de los tres notebooks para que
importen desde src/ en lugar de tener la lógica inline.

Ejecutar desde la raíz del proyecto:
    python scripts/refactor_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOKS = ROOT / 'notebooks'


def load_nb(name: str) -> dict:
    with open(NOTEBOOKS / name, encoding='utf-8') as f:
        return json.load(f)


def save_nb(nb: dict, name: str) -> None:
    with open(NOTEBOOKS / name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  OK {name} guardado')


def find_cell(nb: dict, cell_id: str) -> dict | None:
    for cell in nb['cells']:
        if cell.get('id') == cell_id:
            return cell
    return None


def set_source(cell: dict, code: str) -> None:
    """Reemplaza el source y limpia los outputs del cell."""
    lines = code.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None


# ─────────────────────────────────────────────────────────────────────────────
# 01_eda.ipynb
# ─────────────────────────────────────────────────────────────────────────────

NB01_COLAB_SETUP = """\
# ── Configuración: Colab vs. local ───────────────────────────────────────────
import sys, os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/hey-datathon-2026'
    sys.path.insert(0, BASE)
    DATA_PATH = BASE + '/data/raw/'
    !pip install -q -r /content/drive/MyDrive/hey-datathon-2026/requirements.txt
else:
    BASE = os.path.abspath('..')
    sys.path.insert(0, BASE)
    DATA_PATH = '../data/raw/'

# ── Importar módulos del proyecto ────────────────────────────────────────────
from src.preprocessing import load_data
from src.features import calcular_gaps

print('Entorno   : ' + ('Google Colab' if IN_COLAB else 'local'))
print('DATA_PATH : ' + DATA_PATH)\
"""

NB01_LOAD_DATA = """\
# Carga los tres datasets con tipos correctos (fecha parseada, bools normalizados)
clientes, productos, transacciones = load_data(DATA_PATH)

# El parquet anonimizado no forma parte del flujo de etiquetado
anonimizado = pd.read_parquet(DATA_PATH + 'dataset_50k_anonymized.parquet')

print(f'Clientes     : {clientes.shape}')
print(f'Productos    : {productos.shape}')
print(f'Transacciones: {transacciones.shape}')
print(f'Anonimizado  : {anonimizado.shape}')\
"""

NB01_GAPS = """\
# ── Calcular gaps usando src.features.calcular_gaps ─────────────────────────
# Agrega 'fecha_anterior' y 'gap_dias' al DataFrame de transacciones.
transacciones = calcular_gaps(transacciones)

max_date = transacciones['fecha'].max()

# Clientes con al menos un gap de 45-90 días
users_with_gap = transacciones[
    (transacciones['gap_dias'] >= 45) & (transacciones['gap_dias'] <= 90)
]['user_id'].unique()

# Clientes activos en los últimos 30 días del dataset
last_30_days_limit = max_date - pd.Timedelta(days=30)
users_active_last_30 = transacciones[
    transacciones['fecha'] >= last_30_days_limit
]['user_id'].unique()

# Rescatados = tuvieron gap relevante Y retomaron actividad
rescatados_ids = np.intersect1d(users_with_gap, users_active_last_30)

clientes['segmento'] = 'General'
clientes.loc[clientes['user_id'].isin(rescatados_ids), 'segmento'] = 'Rescatado'

n_rescatados = len(rescatados_ids)
pct_rescatados = (n_rescatados / len(clientes)) * 100
print(f'Clientes Rescatados: {n_rescatados} ({pct_rescatados:.2f}%)')\
"""

NB01_TXN_REGRESO = """\
# 6. Analizar la transacción de regreso
# Primera transacción después del gap de 45-90 días
txn_regreso = transacciones[transacciones['user_id'].isin(rescatados_ids)].copy()
txn_regreso = txn_regreso[txn_regreso['gap_dias'] >= 45].groupby('user_id').head(1)

# Unir con productos para ver el tipo
txn_regreso = txn_regreso.merge(
    productos[['producto_id', 'tipo_producto']], on='producto_id', how='left'
)

# Extraer hora y día de la columna 'fecha' (normalizada por load_data)
txn_regreso['hora_dia']   = txn_regreso['fecha'].dt.hour
txn_regreso['dia_semana'] = txn_regreso['fecha'].dt.day_name()

print("--- Perfil de la Transacción de Regreso ---")
display(txn_regreso[
    ['tipo_operacion', 'tipo_producto', 'categoria_mcc', 'canal', 'dia_semana']
].mode().iloc[0])\
"""


def update_01_eda():
    nb = load_nb('01_eda.ipynb')
    changes = {
        'colab-setup': NB01_COLAB_SETUP,
        'load-data':   NB01_LOAD_DATA,
        'c208d915':    NB01_GAPS,
        '80e2e45e':    NB01_TXN_REGRESO,
    }
    for cell_id, code in changes.items():
        cell = find_cell(nb, cell_id)
        if cell:
            set_source(cell, code)
            print(f'    01_eda  — celda [{cell_id}] actualizada')
        else:
            print(f'    01_eda  — ADVERTENCIA: celda [{cell_id}] no encontrada')
    save_nb(nb, '01_eda.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# 02_churn_labels.ipynb
# ─────────────────────────────────────────────────────────────────────────────

NB02_SETUP = """\
# ── Celda 0: Setup y montaje ────────────────────────────────────────────────
import sys, os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/hey-datathon-2026'
    sys.path.insert(0, BASE)
    DATA_PATH      = BASE + '/data/raw/'
    PROCESSED_PATH = BASE + '/data/processed/'
    !pip install -q pandas numpy matplotlib seaborn
else:
    BASE = os.path.abspath('..')
    sys.path.insert(0, BASE)
    DATA_PATH      = '../data/raw/'
    PROCESSED_PATH = '../data/processed/'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# ── Importar módulos del proyecto ────────────────────────────────────────────
from src.preprocessing import load_data, merge_clientes_etiquetados
from src.features import calcular_gaps, etiquetar_clientes

# ── Paleta visual Hey Banco ───────────────────────────────────────────────────
HEY_COLORS = {
    'healthy':   '#00C48C',
    'recovered': '#FFB347',
    'churned':   '#2D3142',
}
PALETTE = list(HEY_COLORS.values())

sns.set_theme(style='whitegrid', palette=PALETTE, font_scale=1.1)
plt.rcParams.update({'figure.dpi': 110,
                     'axes.spines.top': False,
                     'axes.spines.right': False})

print('Entorno: ' + ('Google Colab' if IN_COLAB else 'local'))
print('DATA_PATH      → ' + DATA_PATH)
print('PROCESSED_PATH → ' + PROCESSED_PATH)\
"""

NB02_CARGA = """\
# ── Celda 1: Carga de datos ─────────────────────────────────────────────────
clientes, productos, transacciones = load_data(DATA_PATH)

# ── Shapes y dtypes ──────────────────────────────────────────────────────────
for nombre, df in [('clientes', clientes), ('productos', productos), ('transacciones', transacciones)]:
    print(f'\\n── {nombre.upper()} ── shape: {df.shape}')
    print(df.dtypes.to_string())

# ── Nulos críticos ───────────────────────────────────────────────────────────
print('\\n── NULOS CRÍTICOS ──')
for nombre, df in [('clientes', clientes), ('transacciones', transacciones)]:
    nulos = df.isnull().sum()
    nulos_criticos = nulos[nulos > 0]
    if nulos_criticos.empty:
        print(f'{nombre}: sin nulos')
    else:
        print(f'{nombre}:\\n{nulos_criticos}')\
"""

NB02_ETIQUETAS = """\
# ── Celda 2: Construcción de etiquetas de churn ─────────────────────────────
# etiquetar_clientes() encapsula la lógica de gaps + asignación de etiquetas.
# Constantes importadas desde src.features:
#   GAP_RECOVERED_MIN=45, GAP_RECOVERED_MAX=90, GAP_CHURNED=90

base = etiquetar_clientes(transacciones)

FECHA_CORTE = transacciones['fecha'].max()
print(f'Fecha de corte del dataset: {FECHA_CORTE.date()}')

# ── Distribución de grupos ───────────────────────────────────────────────────
dist = base['etiqueta'].value_counts().rename_axis('etiqueta').reset_index(name='n')
dist['pct'] = (dist['n'] / dist['n'].sum() * 100).round(1)
print('\\n── DISTRIBUCIÓN DE ETIQUETAS ──')
print(dist.to_string(index=False))

# ── Gráfica de barras ────────────────────────────────────────────────────────
orden = ['healthy', 'recovered', 'churned']
dist_ord = dist.set_index('etiqueta').reindex(orden).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(data=dist_ord, x='etiqueta', y='n',
            palette=[HEY_COLORS[e] for e in orden], ax=axes[0])
axes[0].set_title('Clientes por grupo (n)', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Clientes')
for bar, val in zip(axes[0].patches, dist_ord['n']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')

sns.barplot(data=dist_ord, x='etiqueta', y='pct',
            palette=[HEY_COLORS[e] for e in orden], ax=axes[1])
axes[1].set_title('Distribución porcentual', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('%')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(axes[1].patches, dist_ord['pct']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Distribución de etiquetas de churn — Hey Datathon 2026',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()\
"""

NB02_GUARDAR = """\
# ── Celda 7: Guardar outputs ─────────────────────────────────────────────────
Path(PROCESSED_PATH).mkdir(parents=True, exist_ok=True)

# merge_clientes_etiquetados() une perfil con etiquetas y asigna 'churned'
# a clientes que no tienen transacciones registradas.
clientes_etiquetados = merge_clientes_etiquetados(clientes, base)

print('── VERIFICACIÓN FINAL ──')
print(f'Total clientes etiquetados: {len(clientes_etiquetados):,}')
print(clientes_etiquetados['etiqueta'].value_counts().to_string())
print(f'\\nColumnas en el CSV: {list(clientes_etiquetados.columns)}')

output_path = PROCESSED_PATH + 'clientes_etiquetados.csv'
clientes_etiquetados.to_csv(output_path, index=False, encoding='utf-8')

print(f'\\n✓ Guardado en: {output_path}')
print(f'  Tamaño: {clientes_etiquetados.shape[0]:,} filas × {clientes_etiquetados.shape[1]} columnas')\
"""


def update_02_churn_labels():
    nb = load_nb('02_churn_labels.ipynb')
    changes = {
        'celda-0-setup':    NB02_SETUP,
        'celda-1-carga':    NB02_CARGA,
        'celda-2-etiquetas':NB02_ETIQUETAS,
        'celda-7-guardar':  NB02_GUARDAR,
    }
    for cell_id, code in changes.items():
        cell = find_cell(nb, cell_id)
        if cell:
            set_source(cell, code)
            print(f'    02_churn — celda [{cell_id}] actualizada')
        else:
            print(f'    02_churn — ADVERTENCIA: celda [{cell_id}] no encontrada')
    save_nb(nb, '02_churn_labels.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# 03_modelo.ipynb
# ─────────────────────────────────────────────────────────────────────────────

NB03_SETUP = """\
# ── Celda 0: Setup ────────────────────────────────────────────────────────────
import sys, os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/hey-datathon-2026'
    sys.path.insert(0, BASE)
    DATA_PATH      = BASE + '/data/raw/'
    PROCESSED_PATH = BASE + '/data/processed/'
    MODELS_PATH    = BASE + '/models/'
    !pip install -q xgboost imbalanced-learn shap scikit-learn
else:
    BASE = os.path.abspath('..')
    sys.path.insert(0, BASE)
    DATA_PATH      = '../data/raw/'
    PROCESSED_PATH = '../data/processed/'
    MODELS_PATH    = '../models/'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      StratifiedKFold)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              f1_score, precision_score, recall_score)
import shap

# ── Importar módulos del proyecto ────────────────────────────────────────────
from src.preprocessing import load_data
from src.features import build_model_features
from src.model import save_model, load_model, get_risk_level

# ── Paleta visual Hey Banco (verde / oscuro) ──────────────────────────────────
HEY_GREEN  = '#00C48C'
HEY_DARK   = '#2D3142'
HEY_ORANGE = '#FFB347'

RISK_COLORS  = {'Alto': HEY_DARK,  'Medio': HEY_ORANGE, 'Bajo': HEY_GREEN}
LABEL_COLORS = {'churned': HEY_DARK, 'recovered': HEY_ORANGE, 'healthy': HEY_GREEN}

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print('Entorno      : ' + ('Google Colab' if IN_COLAB else 'local'))
print(f'XGBoost      : {xgb.__version__}')
print(f'DATA_PATH    : {DATA_PATH}')
print(f'PROCESSED    : {PROCESSED_PATH}')
print(f'MODELS_PATH  : {MODELS_PATH}')\
"""

NB03_CARGA = """\
# ── Celda 1: Carga de datos ───────────────────────────────────────────────────

# Dataset con etiquetas generado en 02_churn_labels.ipynb
df_clientes = pd.read_csv(PROCESSED_PATH + 'clientes_etiquetados.csv')

# Transacciones para construir features RFM — load_data normaliza la fecha
_, _, df_tx = load_data(DATA_PATH)

print(f'clientes_etiquetados : {df_clientes.shape}')
print(f'transacciones        : {df_tx.shape}')
print()

# ── Distribución de etiquetas ─────────────────────────────────────────────────
dist     = df_clientes['etiqueta'].value_counts()
dist_pct = df_clientes['etiqueta'].value_counts(normalize=True).mul(100).round(1)

print('── Distribución de etiquetas ──')
for lbl in dist.index:
    print(f'  {lbl:<12} : {dist[lbl]:>5,}  ({dist_pct[lbl]}%)')

# ── Gráfica ───────────────────────────────────────────────────────────────────
orden_lbl = ['healthy', 'recovered', 'churned']
presentes = [l for l in orden_lbl if l in dist.index]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    presentes,
    [dist.get(l, 0) for l in presentes],
    color=[LABEL_COLORS[l] for l in presentes],
    width=0.5, edgecolor='white'
)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
            f'{int(bar.get_height()):,}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Clientes por etiqueta (input del modelo)', fontweight='bold')
ax.set_ylabel('Clientes')
ax.set_xlabel('')
plt.tight_layout()
plt.show()\
"""

NB03_FEATURES = """\
# ── Celda 3: Feature engineering ─────────────────────────────────────────────
# build_model_features() calcula los 8 features RFM y los une con el perfil
# del cliente. Retorna (df_model, FEATURE_COLS) listos para el split.

df_model, FEATURE_COLS = build_model_features(df_clientes, df_tx)

print(f'Dataset de modelado: {df_model.shape}')
print(f'Features ({len(FEATURE_COLS)}):')
for c in FEATURE_COLS:
    null_pct = df_model[c].isnull().mean() * 100
    print(f'  {c:<40}  nulos: {null_pct:.1f}%')\
"""

NB03_SCORES = """\
# ── Celda 8: Score de riesgo para todos los clientes ─────────────────────────

X_all = df_model[FEATURE_COLS].copy()
for col in X_all.select_dtypes(include=['bool']).columns:
    X_all[col] = X_all[col].astype(int)
for col in X_all.select_dtypes(include=['object']).columns:
    X_all[col] = pd.factorize(X_all[col])[0]
X_all = X_all.fillna(X_all.median(numeric_only=True))

df_model['churn_probability'] = best_model.predict_proba(X_all)[:, 1]

# get_risk_level() aplica los umbrales definidos en src/model.py
# (Alto >= 0.70, Medio >= 0.40, Bajo < 0.40)
df_model['risk_level'] = df_model['churn_probability'].apply(get_risk_level)

# ── Distribución ──────────────────────────────────────────────────────────────
risk_dist = df_model['risk_level'].value_counts().reindex(['Alto', 'Medio', 'Bajo'])
risk_pct  = (risk_dist / risk_dist.sum() * 100).round(1)

print('── Distribución de clientes por nivel de riesgo ──')
for nivel in ['Alto', 'Medio', 'Bajo']:
    n = int(risk_dist[nivel])
    p = float(risk_pct[nivel])
    print(f'  {nivel:<8} riesgo : {n:>5,}  ({p:.1f}%)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

niveles = ['Alto', 'Medio', 'Bajo']
valores = [int(risk_dist[n]) for n in niveles]
colores = [RISK_COLORS[n] for n in niveles]
bars = axes[0].bar(niveles, valores, color=colores, width=0.5, edgecolor='white')
for bar, val in zip(bars, valores):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(valores) * 0.01,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Clientes por nivel de riesgo', fontweight='bold')
axes[0].set_ylabel('Clientes')
axes[0].set_xlabel('Nivel de riesgo')

axes[1].hist(df_model['churn_probability'], bins=60,
             color=HEY_GREEN, edgecolor='white', alpha=0.85)
axes[1].axvline(0.40, color=HEY_ORANGE, linestyle='--', lw=2,
                label='Umbral Medio (0.40)')
axes[1].axvline(0.70, color=HEY_DARK, linestyle='--', lw=2,
                label='Umbral Alto (0.70)')
axes[1].set_title('Distribución de probabilidad de churn', fontweight='bold')
axes[1].set_xlabel('P(churn)')
axes[1].set_ylabel('Clientes')
axes[1].legend()

plt.suptitle('Segmentación de riesgo de churn — Hey Banco',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print('\\n── Probabilidad promedio por nivel de riesgo ──')
print(df_model.groupby('risk_level')['churn_probability']
              .agg(['mean', 'min', 'max', 'count'])
              .reindex(['Alto', 'Medio', 'Bajo'])
              .round(3)
              .to_string())\
"""

NB03_GUARDAR = """\
# ── Celda 9: Guardar outputs ──────────────────────────────────────────────────

Path(PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

# ── 1. Guardar modelo con save_model() de src/model.py ───────────────────────
model_path = MODELS_PATH + 'churn_model.pkl'
save_model(best_model, model_path)
print(f'Modelo guardado      : {model_path}')

# ── 2. Guardar scores por cliente ─────────────────────────────────────────────
score_cols     = ['user_id', 'churn_probability', 'risk_level', 'etiqueta']
cols_presentes = [c for c in score_cols if c in df_model.columns]
df_scores      = df_model[cols_presentes].copy()

score_path = PROCESSED_PATH + 'clientes_con_score.csv'
df_scores.to_csv(score_path, index=False, encoding='utf-8')
print(f'Scores guardados     : {score_path}')
print(f'  Shape              : {df_scores.shape}')
print(f'  Columnas           : {df_scores.columns.tolist()}')

# ── 3. Resumen ejecutivo final ────────────────────────────────────────────────
print()
print('══ RESUMEN FINAL DEL MODELO ══')
print(f'  AUC-ROC     : {auc_roc:.4f}')
print(f'  AUC-PR      : {auc_pr:.4f}')
print(f'  F1-Score    : {f1:.4f}')
print(f'  Precision   : {prec:.4f}')
print(f'  Recall      : {rec:.4f}')
print()
print('  Distribución de riesgo (universo completo):')
for nivel in ['Alto', 'Medio', 'Bajo']:
    n_r = int((df_scores['risk_level'] == nivel).sum())
    p_r = n_r / len(df_scores) * 100
    print(f'    {nivel:<8}: {n_r:>5,}  ({p_r:.1f}%)')\
"""


def update_03_modelo():
    nb = load_nb('03_modelo.ipynb')
    changes = {
        'celda-0-setup':    NB03_SETUP,
        'celda-1-carga':    NB03_CARGA,
        'celda-3-features': NB03_FEATURES,
        'celda-8-scores':   NB03_SCORES,
        'celda-9-guardar':  NB03_GUARDAR,
    }
    for cell_id, code in changes.items():
        cell = find_cell(nb, cell_id)
        if cell:
            set_source(cell, code)
            print(f'    03_model — celda [{cell_id}] actualizada')
        else:
            print(f'    03_model — ADVERTENCIA: celda [{cell_id}] no encontrada')
    save_nb(nb, '03_modelo.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Actualizando notebooks...')
    update_01_eda()
    update_02_churn_labels()
    update_03_modelo()
    print('\nRefactor completado.')
