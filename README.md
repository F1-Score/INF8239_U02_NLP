# INF-8239 · Unidad 02 · Proyecto NLP (LAB04, LAB05 y LAB06)

Proyecto base del curso Ciencia de Datos II (autor del proyecto base: Edwin Ramón José Nolasco), ejecutado y documentado por **Emma Carolina Cabrera Valverde**. Clasifica reseñas cortas en inglés como positivas o negativas y explora embeddings de palabras y la estructura de una red.

## Pregunta y corpus

**Pregunta:** ¿se puede predecir si una oración de reseña expresa una opinión positiva o negativa a partir de sus palabras?

**Corpus:** *Sentiment Labelled Sentences* (UCI Machine Learning Repository, Kotzias et al., 2015), licencia CC BY 4.0. Son 3,000 oraciones de reseñas de Amazon (teléfonos y accesorios), IMDb (películas) y Yelp (restaurantes), 500 positivas y 500 negativas por sitio.

- Fuente original: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Ficha completa, diccionario y ejemplos: `docs/DATASET_CARD.md`
- Candidatos comparados: `docs/dataset_candidates.csv`

Los datos no se versionan en Git: cada persona los descarga con el script.

## Instalación

```bash
uv python install 3.12
uv sync
uv run python --version
```

Copie `.env.example` como `.env` y use esta configuración:

```text
DATA_SOURCE=url
DATASET_URL=https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
DATASET_PATH=data/processed/dataset.csv
TEXT_COLUMN=text
TARGET_COLUMN=label
RANDOM_STATE=42
```

## Ejecución

| Laboratorio | Comando | Resultado |
|---|---|---|
| LAB04 · descarga | `uv run python scripts/download_data.py` | ZIP en `data/raw/`, SHA-256 y archivos extraídos |
| LAB04 · preparación | `uv run python scripts/prepare_data.py` | `data/processed/dataset.csv` (text, label, source) |
| LAB04 · auditoría | `uv run python scripts/audit_data.py` | Filas, nulos, duplicados, clases y longitudes |
| LAB05 · entrenamiento | `uv run python scripts/train_text.py` | `reports/text_metrics.csv`, `reports/confusion_text.png`, `reports/error_analysis.csv`, `models/text_model.joblib` |
| LAB05 · errores | `uv run python scripts/categorize_errors.py` | `reports/error_analysis_categorized.csv` y `reports/error_categories.png` (20 errores revisados) |
| LAB05 · aplicación | `uv run streamlit run app/streamlit_app.py` | Clasificador local |
| LAB05 · nube | `uv export --format requirements.txt --output-file requirements-cloud.txt` | Dependencias exportadas |
| LAB06 · embeddings | `uv run python scripts/embeddings_network.py --word service` | `models/word2vec.model`, `reports/word2vec_*.json` |
| LAB06 · experimento | `uv run python scripts/embeddings_network.py --word service --min-count 3` | Segunda configuración para comparar |
| LAB06 · red | `uv run python scripts/embeddings_network.py --word service --network-demo` | `reports/centralities.csv`, `reports/network.png`, `reports/social_network.graphml` |

`train_text.py` elimina las oraciones duplicadas antes de partir los datos (75 % entrenamiento, 25 % prueba, estratificado, semilla 42) y compara `DummyClassifier`, `ComplementNB` y `LogisticRegression`, cada uno con TF-IDF dentro de su pipeline para evitar fuga de información. Guarda el pipeline con mayor F1 macro.

## Resultados principales

| Laboratorio | Resultado |
|---|---|
| LAB04 | 3,000 oraciones, 0 nulos, 18 duplicados, clases 50/50 |
| LAB05 | Naive Bayes F1 macro 0.819 (regresión logística 0.788, baseline 0.333); 135 errores en 746 oraciones de prueba |
| LAB06 | "service": vocabulario 5,155 → 1,468 al pasar `min_count` de 1 a 3; red de 34 nodos, 78 aristas, 3 comunidades, modularidad 0.411 |

## Pruebas

```bash
uv run pytest -q
```

## Documentación

- `docs/DATASET_CARD.md`: procedencia, licencia, diccionario, ejemplos, sesgos y usos no permitidos.
- `reports/lab05_errores.md` y `reports/lab06_embeddings_red.md`: interpretación de resultados.
- `notebooks/Ejercicio03_NLP.ipynb`: resultados consolidados.
- `AI_USE_DECLARATION.md`: asistencia de IA y verificaciones.

## Referencias

- Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). From group to individual labels using deep features. *Proceedings of KDD 2015*, 597–606. https://doi.org/10.1145/2783258.2783380
- Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. *Journal of Anthropological Research, 33*(4), 452–473.
