# Urban Agent — Agente conversacional multimodal para quejas urbanas

Sistema de análisis automático de incidencias ciudadanas basado en técnicas de Procesamiento del Lenguaje Natural (PLN). El sistema integra clasificación temática no supervisada, análisis de sentimiento con modelos contextuales, recuperación semántica y un agente conversacional basado en LLM, aplicado sobre datos reales del portal de participación ciudadana del Ayuntamiento de Madrid.

**Autora:** Salma Ghailan Serroukh  
**Contexto:** Máster en Inteligencia Artificial — Módulo de Procesamiento del Lenguaje Natural  
**Memoria técnica:** [docs/memoria_urban_agent.pdf](docs/memoria_urban_agent.pdf)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat-square)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?style=flat-square)
![License](https://img.shields.io/badge/Licencia-Académica-lightgrey?style=flat-square)

---

## Descripción del sistema

El proyecto aborda el problema del análisis automático de quejas urbanas en texto libre, un dominio caracterizado por alta variabilidad léxica, lenguaje informal y expresiones implícitas de malestar. El sistema implementa un pipeline completo que transforma texto ciudadano no estructurado en información estructurada y accionable para la gestión municipal.

El pipeline integra cinco componentes:

1. **Preprocesamiento lingüístico** — limpieza, normalización y lematización con spaCy sobre corpus en español
2. **Clasificación temática no supervisada** — NMF sobre TF-IDF y LDA sobre Bag of Words, con etiquetado semiautomático mediante perfiles léxicos (human-in-the-loop)
3. **Análisis de sentimiento** — transfer learning con BETO (BERT en español), fine-tuned para clasificación de polaridad y estimación de intensidad de malestar
4. **Recuperación semántica** — embeddings Word2Vec con similitud coseno para validación cualitativa y búsqueda de incidencias similares
5. **Agente conversacional** — LLM que integra las señales anteriores para generar respuestas empáticas, estimar prioridad de intervención y detectar información faltante

---

## Dataset

Los datos proceden del portal de **Datos Abiertos del Ayuntamiento de Madrid**, plataforma de participación ciudadana Decide Madrid.

- **Fuente:** [https://datos.madrid.es](https://datos.madrid.es)
- **Formato:** CSV con 33.671 registros y múltiples variables textuales
- **Contenido:** propuestas y quejas ciudadanas sobre gestión urbana y servicios públicos

El dataset **no se incluye en el repositorio**. Para descargarlo:

```bash
python scripts/download_data.py
```

Esto genera: `data/raw/madrid_decide.csv`

---

## Estructura del repositorio

```
urban-complaints-nlp/
│
├── notebooks/
│   ├── 01_clasificacion_tematica.ipynb     # pipeline temático completo
│   ├── 02_analisis_sentimiento.ipynb       # BETO + Naive Bayes
│   └── 03_descripcion_imagenes.ipynb       # BLIP image captioning (opcional)
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py                 # limpieza y normalización
│   │   └── spacy_processor.py              # lematización y stopwords
│   │
│   ├── topic_modeling/
│   │   ├── __init__.py
│   │   ├── modelos.py                      # carga de artefactos entrenados
│   │   └── topic_predict.py                # inferencia temática (NMF + LDA)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── similitud_semantica_iu.py       # Word2Vec + similitud coseno + UI
│   │
│   └── agent/
│       ├── __init__.py
│       └── agente_urbano.py                # orquestador principal del agente
│
├── scripts/
│   └── download_data.py
│
├── data/
│   ├── raw/                                # dataset original (no versionado)
│   ├── processed/                          # CSVs enriquecidos (no versionados)
│   └── imagenes/                           # imágenes de ejemplo para la demo
│       ├── basura.avif
│       └── parque_bonito.jpg
│
├── artefactos/                             # modelos entrenados (no versionados)
├── figures/                                # visualizaciones generadas por los notebooks
├── outputs/                                # salidas del agente en ejecución (no versionadas)
├── docs/
│   └── memoria_urban_agent.pdf
│
├── app.py                                  # demo interactiva con Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

### Qué se versiona y qué no

| Carpeta | Versionada | Motivo |
|---------|-----------|--------|
| `src/`, `notebooks/`, `scripts/` | Sí | código fuente |
| `figures/` | Sí | visualizaciones del trabajo |
| `data/imagenes/` | Sí | imágenes de ejemplo para la demo |
| `docs/` | Sí | memoria técnica |
| `data/raw/`, `data/processed/` | No | datos pesados o reproducibles |
| `artefactos/` | No | modelos reproducibles ejecutando los notebooks |
| `outputs/` | No | resultados de ejecución |
| `venv_test/` | No | entorno virtual |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/urban-complaints-nlp.git
cd urban-complaints-nlp
```

### 2. Crear entorno virtual

```bash
python -m venv venv_test
source venv_test/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar modelo de spaCy en español

```bash
python -m spacy download es_core_news_sm
```

### 5. Configurar la clave de API del LLM

El agente utiliza un LLM externo. Se recomienda **Groq** (tier gratuito disponible).

**Obtener clave de Groq:**

1. Crear cuenta en [https://console.groq.com](https://console.groq.com)
2. Menú izquierdo → API Keys → Create API Key
3. Copiar la clave inmediatamente (solo se muestra una vez)

**Configurar como variable de entorno** (la clave nunca queda en el código ni en el repositorio):

```bash
echo 'export GROQ_API_KEY="gsk_..."' >> ~/.bashrc
source ~/.bashrc
```

Verificar:

```bash
echo $GROQ_API_KEY
```

> Si prefieres usar OpenAI, exporta `OPENAI_API_KEY` de la misma forma y consulta la sección de configuración avanzada.

---

## Ejecución del pipeline

Los notebooks deben ejecutarse en orden. Cada uno lee el output del anterior y genera nuevos artefactos y datos procesados.

### Notebook 01 — Clasificación temática

```
notebooks/01_clasificacion_tematica.ipynb
```

**Entrada:** `data/raw/madrid_decide.csv`

**Qué hace:**
- Preprocesamiento: limpieza HTML, normalización, lematización con spaCy
- Vectorización: TF-IDF (para NMF) y Bag of Words (para LDA)
- Modelado temático: NMF con k=15 temas, LDA con k=20 temas
- Etiquetado semiautomático de temas mediante perfiles léxicos definidos manualmente
- Validación cualitativa con Word2Vec y similitud coseno
- Visualización t-SNE de los espacios documento-tópico

**Genera en `artefactos/`:**
```
tfidf.joblib
nmf.joblib
bow.joblib
lda.joblib
nombres_temas_auto.json      # etiquetas NMF
nombres_temas_lda.json       # etiquetas LDA
```

**Genera en `data/processed/`:**
```
df_sugerencias_controlada_temas.csv
```

**Genera en `figures/`:**
```
tsne_nmf.png
tsne_lda.png
tsne_lsi.png
similitud_documento_existente.png
similitud_documento_nuevo.png
```

---

### Notebook 02 — Análisis de sentimiento

```
notebooks/02_analisis_sentimiento.ipynb
```

**Entrada:** `data/processed/df_sugerencias_controlada_temas.csv`

**Qué hace:**
- Inferencia por lotes con BETO (`finiteautomata/beto-sentiment-analysis`)
- Clasificación de polaridad: NEG / NEU / POS
- Cálculo de intensidad de malestar en 4 niveles: neutro, ligeramente negativo, negativo, muy negativo
- Análisis cruzado temática x polaridad por categoría
- Comparativa metodológica con clasificador Naive Bayes clásico

**Genera en `data/processed/`:**
```
df_sugerencias_controlada_temas_beto_sentimiento.csv
```

**Genera en `figures/`:**
```
distribucion_polaridad_beto.png
distribucion_intensidad_emocional_beto.png
polaridad_por_tematica_lda.png
```

---

### Notebook 03 — Descripción de imágenes (opcional)

```
notebooks/03_descripcion_imagenes.ipynb
```

Exploración del módulo de image captioning con BLIP (`Salesforce/blip-image-captioning-base`). Módulo complementario al pipeline principal, activo solo cuando se proporciona imagen.

---

## Demo interactiva

```bash
streamlit run app.py
```

La aplicación permite:
- Introducir una incidencia urbana en texto libre
- Adjuntar una imagen opcional del problema
- Recibir la respuesta del agente con temática detectada, nivel de malestar y prioridad sugerida
- Inspeccionar el JSON intermedio completo con el contexto analítico y la salida del LLM

---

## Ejecutar el agente directamente

```bash
python src/agent/agente_urbano.py
```

Ejecuta el agente con el mensaje y la imagen de prueba definidos en el bloque `__main__`. La salida se guarda automáticamente en `outputs/agente_output.json`.

---

## Arquitectura del agente

```
Texto ciudadano
      │
      ├──► Preprocesamiento (text_cleaner + spacy_processor)
      │
      ├──► Clasificación temática
      │         ├── NMF + TF-IDF  →  tema NMF + confianza + top-k
      │         └── LDA + BoW     →  tema LDA + probabilidad + top-k
      │
      ├──► Análisis de sentimiento
      │         └── BETO          →  NEG/NEU/POS + intensidad de malestar
      │
      ├──► Descripción visual (opcional)
      │         └── BLIP          →  caption en inglés
      │
      └──► LLM (Groq / OpenAI)
                └── JSON estructurado:
                      estado
                      respuesta_ciudadano
                      prioridad (baja / media / alta)
                      datos_faltantes
                      resumen_para_supervision
```

---

## Tecnologías

| Categoría | Herramienta |
|-----------|-------------|
| Lenguaje | Python 3.12 |
| Preprocesado | pandas, spaCy `es_core_news_sm` |
| Vectorización | scikit-learn (TF-IDF, BoW) |
| Topic modeling | scikit-learn (NMF), gensim (LDA, LSI) |
| Embeddings | gensim Word2Vec |
| Sentimiento | transformers, `finiteautomata/beto-sentiment-analysis` |
| Image captioning | transformers, `Salesforce/blip-image-captioning-base` |
| LLM | Groq API `llama-3.3-70b-versatile` / OpenAI API |
| Demo | Streamlit |
| Visualización | matplotlib, seaborn, plotly |

---

## Configuración avanzada

### Cambiar de Groq a OpenAI

En `src/agent/agente_urbano.py`, sustituir:

```python
from groq import Groq
client = Groq()
```

por:

```python
from openai import OpenAI
client = OpenAI()
```

Y cambiar el modelo por defecto:

```python
llm_model: str = "gpt-4o-mini"
```

Exportar la clave:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### Activar el módulo visual (BLIP)

Por defecto el módulo visual está desactivado para reducir tiempo de carga. Para activarlo programáticamente:

```python
import sys
sys.path.insert(0, "src")

from agent.agente_urbano import agente_urbano, VisionConfig, load_topic_models
from PIL import Image

topic_models = load_topic_models("artefactos")
vision_cfg = VisionConfig(enabled=True)
image = Image.open("data/imagenes/basura.avif").convert("RGB")

out = agente_urbano(
    mensaje="Hay basura acumulada en la calle",
    topic_models=topic_models,
    image=image,
    vision_cfg=vision_cfg,
)
```

---

## Reproducibilidad

Los artefactos (`artefactos/`) y los datos procesados (`data/processed/`) no se incluyen en el repositorio porque son completamente reproducibles ejecutando los notebooks en orden.

Secuencia completa de reproducción:

```
1. python scripts/download_data.py
        → data/raw/madrid_decide.csv

2. notebooks/01_clasificacion_tematica.ipynb
        → artefactos/tfidf.joblib
        → artefactos/nmf.joblib
        → artefactos/bow.joblib
        → artefactos/lda.joblib
        → artefactos/nombres_temas_auto.json
        → artefactos/nombres_temas_lda.json
        → data/processed/df_sugerencias_controlada_temas.csv
        → figures/tsne_nmf.png, tsne_lda.png, ...

3. notebooks/02_analisis_sentimiento.ipynb
        → data/processed/df_sugerencias_controlada_temas_beto_sentimiento.csv
        → figures/distribucion_polaridad_beto.png, ...

4. streamlit run app.py
   o
   python src/agent/agente_urbano.py
        → outputs/agente_output.json
```

---

## Licencia

Proyecto desarrollado con fines académicos y demostrativos. No destinado a uso comercial.
