# Declaración de uso de IA · Proyecto NLP (LAB04, LAB05 y LAB06)

**Estudiante:** Emma Carolina Cabrera Valverde
**Asignatura:** INF-8239 · Ciencia de Datos II
**Herramienta:** Claude (Anthropic), asistente conversacional con acceso a los archivos del proyecto

## Procedencia del código
- El proyecto base (`src/inf8239_u02`, `audit_data.py`, `train_text.py`, `embeddings_network.py`, aplicación, pruebas, `pyproject.toml` y `uv.lock`) es material del profesor Edwin Ramón José Nolasco, descargado del aula virtual.
- Se utilizó como referencia el material compartido en el curso; la ejecución, los resultados y la redacción de esta entrega son propios. El corpus y las decisiones de análisis también son propios.

## En qué ayudó la IA
1. Buscar y verificar un corpus público que cumpliera los criterios del manual (fuente, licencia, etiqueta, acceso reproducible, ausencia de datos personales) y comparar dos candidatos.
2. Ampliar `download_data.py` para descargar y extraer un ZIP, y escribir `prepare_data.py` para unir los tres archivos del corpus. Durante la auditoría detectó el carácter U+0085 que partía dos oraciones de IMDb y corrigió la lectura.
3. Ajustar `train_text.py` para guardar el modelo con mejor F1 macro (antes fijo en regresión logística), registrar accuracy y recall por clase, añadir la fuente de cada error y usar colores pastel.
4. Añadir a `embeddings_network.py` las opciones `--window` y `--min-count` y un registro JSON de cada corrida, y rediseñar el gráfico de la red.
5. Proponer una categorización de 20 errores en `categorize_errors.py`, que revisé.
6. Redactar borradores del README, Dataset Card, informes de `reports/`, notebook e informe Word a partir de mis resultados.

## Qué hice y verifiqué yo
- Ejecuté todo en mi computadora: instalación con `uv`, descarga (SHA-256 registrado), preparación, auditoría, entrenamiento, aplicación Streamlit, exportación de dependencias, embeddings, red y pruebas (6 aprobadas).
- Probé la aplicación con una oración clara, una negación y una frase en español, y comprobé la advertencia de dominio.
- Revisé que las categorías de los errores correspondieran a cada oración y contrasté las cifras de los documentos con los archivos de `reports/`.

## Límites
La IA reprodujo partes del flujo en un entorno aparte para verificar los scripts, pero todos los resultados reportados provienen de las ejecuciones en mi computadora. Asumo la responsabilidad del código, de la verificación de los resultados y de las conclusiones.
