# LAB06 · Embeddings y análisis estructural de una red

## 1. Word2Vec sobre el corpus de reseñas

Configuración fija: 60 dimensiones, 30 épocas, semilla 42, `workers=1`. Palabra estudiada: **service** (108 apariciones en 33,698 tokens). Resultados en `word2vec_service_w5_mc1.json` y `word2vec_service_w5_mc3.json`.

| Configuración | Vocabulario | Cobertura | Vecinos más cercanos de "service" |
|---|---:|---:|---|
| `window=5`, `min_count=1` (base) | 5,155 | 1.000 | impeccable 0.951, negligent 0.937, food 0.932, server 0.929, customer 0.929, attentive 0.928, outstanding 0.925, downside 0.921, boot 0.921, personable 0.918 |
| `window=5`, `min_count=3` (experimento) | 1,468 | 0.868 | attentive 0.910, friendly 0.902, customer 0.897, food 0.880, staff 0.872, server 0.867, slow 0.864, selection 0.861, terrible 0.859, extremely 0.849 |

**Hipótesis.** Con `min_count=1`, palabras que aparecen una o dos veces tienen vectores poco entrenados y pueden quedar cerca de cualquier término por azar. Subir `min_count` a 3 debería eliminar ese ruido a cambio de dejar parte del texto fuera del vocabulario.

**Resultado.** El vocabulario baja de 5,155 a 1,468 palabras (−72 %) y la cobertura de 100 % a 86.8 %: 3,687 palabras raras, que suman el 13.2 % de los tokens, quedan fuera. Con la base, la mitad de los vecinos de "service" son palabras que aparecen solo una o dos veces ("negligent" y "downside" 1 vez, "impeccable" 2 veces, "personable" 1 vez). Con `min_count=3`, los vecinos son palabras frecuentes del mismo contexto de restaurante: "attentive" (7 apariciones), "friendly" (29), "staff", "slow" (20) y "terrible" (32).

**Interpretación.** El cambio no modifica la ventana de contexto, sino qué palabras tienen evidencia suficiente para un vector confiable. Los vecinos indican que "service" aparece junto a descripciones de la atención, tanto positivas ("attentive", "friendly") como negativas ("slow", "terrible", "negligent"): Word2Vec captura el tema del contexto, no la polaridad. Una similitud de 0.95 con "impeccable" no significa que sean sinónimos; significa que en este corpus pequeño se usan en oraciones parecidas.

**Decisión.** Para describir el vocabulario frecuente del corpus se prefiere `min_count=3`. El modelo guardado en `models/word2vec.model` se deja con la configuración base del proyecto (`window=5`, `min_count=1`) para que la red y las pruebas se reproduzcan igual, y ambos resultados quedan documentados.

## 2. Red de demostración: club de karate de Zachary

**Qué representa.** Cada **nodo** es un miembro de un club de karate universitario estudiado por Zachary (1977). Cada **arista** indica que dos miembros interactuaban fuera del club según las observaciones del autor. La red no es dirigida ni tiene pesos en esta versión. **No incluye** la intensidad ni el tipo de la relación, el periodo exacto, atributos de las personas ni su consentimiento para un análisis nuevo.

| Indicador | Valor |
|---|---:|
| Nodos | 34 |
| Aristas | 78 |
| Densidad | 0.139 |
| Comunidades (modularidad codiciosa) | 3 (17, 9 y 8 nodos) |
| Modularidad | 0.411 |

Solo el 13.9 % de las conexiones posibles existe, y la partición en tres grupos tiene modularidad 0.411: hay bastantes más aristas dentro de cada grupo que entre grupos.

## 3. Dos métricas comparadas: grado frente a intermediación

Además de los dos nodos con más conexiones (0 y 33, primeros en grado, intermediación y PageRank), la comparación más informativa está en dos nodos intermedios:

| Nodo | Conexiones | Grado (puesto) | Intermediación (puesto) | PageRank (puesto) |
|---:|---:|---:|---:|---:|
| 1 | 9 | 0.273 (5.º) | 0.054 (7.º) | 0.057 (5.º) |
| 31 | 6 | 0.182 (6.º) | 0.138 (5.º) | 0.042 (6.º) |

El nodo 1 tiene más conexiones directas, pero la mayoría (6 de 9) dentro de su propio grupo y casi todas hacia nodos que ya están conectados entre sí: su intermediación es baja. El nodo 31 tiene menos conexiones, pero une a cinco miembros de la comunidad más grande con el nodo 0, centro de otro grupo, por lo que aparece en muchos caminos más cortos: su intermediación es 2.5 veces la del nodo 1. Es el caso de "grado moderado y betweenness alta" del manual: un **puente estructural**.

**Interpretación permitida.** El nodo 31 ocupa una posición de conexión entre grupos en esta red; si se eliminara, varios caminos entre comunidades serían más largos.

**Interpretación que no puede sostenerse.** No se puede afirmar que el nodo 31 sea "la persona más influyente", que controle la información o que tenga amistad con quienes conecta. La centralidad mide posición en el grafo, no influencia real ni causalidad. Tampoco se puede decir que las tres comunidades sean grupos sociales reales: son el resultado de un algoritmo sobre esta red.

## 4. Siguiente experimento
- Embeddings: entrenar con un corpus mayor o usar vectores preentrenados y comparar si los vecinos de "service" se separan por polaridad.
- Red: comparar la partición del algoritmo con otra técnica (por ejemplo, Louvain) y medir cuánto cambia la asignación de los nodos puente.

## Referencia
Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. *Journal of Anthropological Research, 33*(4), 452–473.
