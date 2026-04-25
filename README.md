# Hey Datathon 2026 — Motor de Inteligencia & Atención Personalizada

> **Reto Hey 2026** · Hey Banco · Datathon Nacional de Data Science e Inteligencia Artificial

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmateo101/hey-datathon-2026/blob/main/notebooks/01_eda.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Descripcion del Reto

El **Reto Hey 2026** desafía a los equipos a construir un motor de inteligencia personalizada para los clientes de Hey Banco. El objetivo es analizar el comportamiento financiero de los usuarios y diseñar soluciones de IA que mejoren la experiencia, la atención y las recomendaciones personalizadas de productos financieros.

---

## Datasets

Los datasets deben colocarse en `data/raw/` localmente **o** en Google Drive (ver sección Colab).

| Archivo | Descripcion | Formato |
|---|---|---|
| `hey_clientes.csv` | Perfil demografico y segmentacion de clientes | CSV |
| `hey_productos.csv` | Catalogo de productos financieros Hey Banco | CSV |
| `hey_transacciones.csv` | Historial de transacciones por cliente | CSV |
| `dataset_50k_anonymized.parquet` | Dataset anonimizado de 50k registros | Parquet |

> **Nota:** Los datos crudos estan en `.gitignore` y no se suben al repositorio por razones de privacidad y tamano.

---

## Notebooks

| Notebook | Descripcion | Abrir en Colab |
|---|---|---|
| `01_eda.ipynb` | Analisis Exploratorio de Datos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmateo101/hey-datathon-2026/blob/main/notebooks/01_eda.ipynb) |
| `02_features.ipynb` | Feature Engineering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmateo101/hey-datathon-2026/blob/main/notebooks/02_features.ipynb) |
| `03_modelo.ipynb` | Entrenamiento y evaluacion del modelo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmateo101/hey-datathon-2026/blob/main/notebooks/03_modelo.ipynb) |
| `04_demo.ipynb` | Demo final del motor personalizado | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmateo101/hey-datathon-2026/blob/main/notebooks/04_demo.ipynb) |

---

## Estructura del Proyecto

```
hey-datathon-2026/
├── data/
│   ├── raw/                  # Datasets originales (ignorados por git)
│   └── processed/            # Datos limpios y transformados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modelo.ipynb
│   └── 04_demo.ipynb
├── src/
│   ├── preprocessing.py      # Funciones de limpieza de datos
│   ├── features.py           # Feature engineering
│   └── model.py              # Logica del modelo
├── models/                   # Modelos entrenados (.pkl, .joblib)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalacion Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Mmateo101/hey-datathon-2026.git
cd hey-datathon-2026

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Agregar los datasets en data/raw/
# Copiar: hey_clientes.csv, hey_productos.csv,
#         hey_transacciones.csv, dataset_50k_anonymized.parquet

# 5. Lanzar Jupyter
jupyter notebook
```

---

## Uso en Google Colab

### Montar Google Drive y cargar datos

Al inicio de cada notebook en Colab, ejecuta la siguiente celda:

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/hey-datathon-2026/src')

DATA_PATH = '/content/drive/MyDrive/hey-datathon-2026/data/raw/'
```

### Instalar dependencias en Colab

```python
!pip install -q -r /content/drive/MyDrive/hey-datathon-2026/requirements.txt
```

---

## Stack Tecnologico

| Categoria | Tecnologias |
|---|---|
| Lenguaje | Python 3.10+ |
| Analisis de datos | pandas, numpy |
| Visualizacion | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn |
| Embeddings / NLP | sentence-transformers |
| IA Generativa | openai, anthropic |
| Formatos de datos | pyarrow, fastparquet |
| Entorno | Jupyter, Google Colab |
| Variables de entorno | python-dotenv |

---

## Variables de Entorno

Crea un archivo `.env` en la raiz (nunca lo subas a git):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Cargalas en tu notebook:

```python
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
```

---

## Equipo

| Nombre | Rol |
|---|---|
| Mateo | Data Scientist / ML Engineer |

---

*Desarrollado para el Reto Hey 2026 — Hey Banco*
