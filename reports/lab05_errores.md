# LAB05 · Clasificación de sentimiento y análisis de errores

## Protocolo
- Corpus: 3,000 oraciones; se eliminan 18 duplicados antes de partir (quedan 2,982).
- Partición estratificada única: 75 % entrenamiento (2,236) y 25 % prueba (746: 373 negativas y 373 positivas), semilla 42.
- Tres pipelines con TF-IDF de unigramas y bigramas dentro de cada pipeline (sin fuga): `DummyClassifier` (clase más frecuente), `ComplementNB` y `LogisticRegression` balanceada.

## Resultados (`text_metrics.csv`)

| Modelo | F1 macro | Accuracy | Recall negative | Recall positive |
|---|---:|---:|---:|---:|
| Naive Bayes | **0.819** | **0.819** | 0.812 | 0.826 |
| Regresión logística | 0.788 | 0.788 | 0.783 | 0.794 |
| Dummy | 0.333 | 0.500 | 1.000 | 0.000 |

El baseline predice siempre la misma clase: 50 % de accuracy, pero recall 0 para `positive`. Los dos modelos reales lo superan por mucho. **Naive Bayes es el mejor en F1 macro y en el recall de ambas clases**, por lo que se seleccionó y se guardó en `models/text_model.joblib`. El proyecto base guardaba siempre la regresión logística; se cambió para elegir por F1 macro, que es el criterio del laboratorio.

## Matriz de confusión (Naive Bayes, `confusion_text.png`)

| | Predice negative | Predice positive |
|---|---:|---:|
| **Real negative** | 303 | 70 |
| **Real positive** | 65 | 308 |

611 aciertos de 746. La celda fuera de la diagonal más grande son las **70 reseñas negativas clasificadas como positivas** (falsos positivos). En un uso real, por ejemplo para detectar clientes insatisfechos, esas quejas pasarían desapercibidas; es el error más costoso. La clase con menor recall es `negative` (0.812).

## 20 errores categorizados (`error_analysis_categorized.csv`)
El modelo comete 135 errores en la prueba. Se revisó una muestra aleatoria de 20 (semilla 8239) para no elegir casos a conveniencia.

| Categoría | Errores | Ejemplo |
|---|---:|---|
| Sentimiento implícito | 6 | "My father has the V265, and the battery is dying." (negativa, predicha positiva) |
| Negación | 4 | "The soundtrack wasn't terrible, either." (positiva, predicha negativa) |
| Contraste de opiniones | 4 | "It was a pale color instead of nice and char and has NO flavor." |
| Vocabulario poco frecuente | 3 | "Best fish I've ever had in my life!" ("fish" aparece en 4 reseñas negativas y 2 positivas) |
| Ironía | 1 | "I felt asleep the first time I watched it, so I can recommend it for insomniacs." |
| Texto insuficiente | 1 | "Lousy product." |
| Lenguaje informal | 1 | "OMG I felt like I had never eaten Thai food until this dish." |

**Interpretación.** El 70 % de los errores revisados (implícito, negación y contraste) comparte una causa: TF-IDF mira palabras sueltas o pares de palabras, no el sentido de la oración. Una queja dicha como hecho, un "not bad" o una frase con dos opiniones confunden al modelo. La prueba en la aplicación lo confirma: "The movie was not bad at all." se clasificó como `negative`.

**Hipótesis para el siguiente experimento.** Marcar las palabras que siguen a una negación (por ejemplo, convertir "not bad" en "not NOT_bad") o usar un modelo que lea la oración completa (embeddings contextuales) debería reducir los errores de negación y contraste. Se mediría con la misma partición y la misma muestra de errores.
