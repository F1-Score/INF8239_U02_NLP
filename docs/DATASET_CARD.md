# Dataset Card · Sentiment Labelled Sentences

## Identificación
- **Nombre:** Sentiment Labelled Sentences
- **Fuente original:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- **Responsable:** Dimitrios Kotzias (donado el 29 de mayo de 2015), creado para Kotzias, Denil, de Freitas y Smyth (2015)
- **Versión o fecha:** ZIP descargado el 31 de octubre de 2026; 84,188 bytes; SHA-256 `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0). Permite descargar, analizar y publicar resultados derivados citando la fuente.
- **Idioma:** Inglés

## Propósito y variable objetivo
Pregunta del proyecto: ¿se puede predecir si una oración de reseña expresa una opinión positiva o negativa a partir de sus palabras? La variable objetivo es `label` (`negative` o `positive`). El uso previsto es académico: comparar clasificadores de texto y estudiar sus errores.

## Diccionario de datos

Archivo procesado: `data/processed/dataset.csv` (generado por `scripts/prepare_data.py`).

| Columna | Tipo | Descripción | Valores o unidad |
|---|---|---|---|
| `text` | texto | Oración extraída de una reseña en línea | 1 a 71 palabras; 5 a 477 caracteres |
| `label` | categórica | Polaridad asignada por los autores del corpus | `negative` (puntuación 0) o `positive` (puntuación 1) |
| `source` | categórica | Sitio de origen de la reseña | `amazon` (teléfonos y accesorios), `imdb` (películas), `yelp` (restaurantes) |

## Procedimiento de obtención
1. `.env` con `DATA_SOURCE=url` y la URL del ZIP de UCI (ver `README.md`).
2. `uv run python scripts/download_data.py`: descarga el ZIP a `data/raw/`, imprime su SHA-256 y extrae los archivos sin modificarlos.
3. `uv run python scripts/prepare_data.py`: une los tres archivos `*_labelled.txt` (formato `oración<TAB>puntuación`) en un CSV.

Transformaciones aplicadas:
- La puntuación 0/1 se traduce a `negative`/`positive` para que la etiqueta sea legible.
- Se agrega la columna `source` con el nombre del sitio.
- Dos oraciones de IMDb contienen el carácter U+0085 (puntos suspensivos de Windows-1252). Leerlas con `splitlines()` las partía en dos líneas sin etiqueta; el script separa solo por salto de línea real y reemplaza ese carácter por `...`. Resultado: 3,000 oraciones leídas, 0 descartadas.
- No se corrigen faltas de ortografía ni se cambia el texto de otra forma. Los duplicados se conservan en el CSV y se eliminan dentro de `train_text.py`, antes de partir los datos.

## Calidad observada
Auditoría con `scripts/audit_data.py`:

| Indicador | Valor |
|---|---|
| Filas y columnas | 3,000 × 3 |
| Nulos en `text` / `label` | 0 / 0 |
| Textos duplicados | 18 (todos con la misma etiqueta en cada copia) |
| Distribución de clases | 50 % negative, 50 % positive |
| Por fuente | 500 positivas y 500 negativas en cada sitio |
| Longitud en caracteres (p50 / p90 / p99 / máx.) | 55 / 121 / 213 / 477 |
| Mediana de caracteres por fuente | Amazon 48, Yelp 51, IMDb 66 |

Las clases están balanceadas por diseño, así que accuracy y F1 macro casi coinciden. Los duplicados son frases cortas y frecuentes (por ejemplo, "Works great." o "I won't be back."); si quedaran a la vez en entrenamiento y prueba inflarían el resultado, por eso se eliminan antes de partir. Hay textos muy cortos (desde 5 caracteres, como "10/10") que dan poca información al modelo.

### Cinco ejemplos
Los textos no contienen nombres de usuario ni datos personales; se muestran tal como aparecen en el corpus.

| text | label | source |
|---|---|---|
| Those phones are working just fine now. | positive | amazon |
| This product is great... it makes working a lot easier I can go to the copier while waiting on hold for something. | positive | amazon |
| Let's start with all the problemsthe acting, especially from the lead professor, was very, very bad. | negative | imdb |
| It failed to convey the broad sweep of landscapes that were a great part of the original. | negative | imdb |
| You'd have to have the IQ of particularly stupid mollusk not to see that one coming. | negative | imdb |

## Población cubierta y excluida
- **Cubierta:** oraciones en inglés de reseñas públicas de Amazon (teléfonos y accesorios), IMDb (películas) y Yelp (restaurantes). Según el `readme.txt` del corpus, se eligieron al azar entre conjuntos más grandes, buscando oraciones con polaridad clara y evitando las neutrales.
- **Excluida:** opiniones neutrales o mixtas, otros idiomas (incluido el español), otros productos y servicios, y cualquier información sobre quién escribió la reseña o cuándo.
- No hay fechas: no es posible saber a qué periodo representan las opiniones.

## Riesgos, sesgos y usos prohibidos
- **Sesgo de selección:** al quitar las oraciones neutrales, el problema es más fácil que en reseñas reales; el desempeño fuera de este corpus será menor.
- **Desequilibrio de dominios:** solo tres tipos de producto. Palabras como "battery" o "fish" pueden asociarse a una polaridad por las pocas reseñas que las contienen.
- **Idioma:** el modelo no entiende español; ante un texto en español responde casi al azar (comprobado en la aplicación con "Excelente servicio").
- **Etiquetas:** fueron asignadas por los autores del corpus; algunas oraciones aisladas pierden el contexto de la reseña completa.
- **Usos prohibidos:** evaluar o sancionar a personas, empleados o negocios concretos; moderar contenido de forma automática; decidir sobre reseñas en otros idiomas o dominios; presentar la predicción como opinión verificada.

## Cita
Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). From group to individual labels using deep features. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 597–606. https://doi.org/10.1145/2783258.2783380
