"""LAB05 · Categorización manual de 20 errores del modelo seleccionado.

Toma una muestra reproducible de 20 errores de reports/error_analysis.csv
(semilla 8239, sin elegir casos a mano) y les asigna la categoría revisada por
la estudiante. Las categorías se guardan por identificador de fila del corpus,
no por posición, para que no se desalineen si cambia el orden del archivo.

Salidas:
- reports/error_analysis_categorized.csv: los 20 errores con categoría y nota.
- reports/error_categories.png: frecuencia de cada categoría.

    uv run python scripts/categorize_errors.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from inf8239_u02.config import ROOT

REPORTS = ROOT / "reports"
SAMPLE_SEED = 8239

# fila del corpus -> (categoría, nota de la revisión)
REVIEW = {
    995: ("Sentimiento implícito", "Describe un defecto (pantalla que se ensucia) sin palabras de opinión."),
    908: ("Sentimiento implícito", "Elogia el volumen describiendo un uso concreto, sin adjetivos positivos."),
    833: ("Sentimiento implícito", "'the battery is dying' es una queja expresada como hecho."),
    2597: ("Sentimiento implícito", "La intención de volver expresa satisfacción sin palabras positivas."),
    302: ("Sentimiento implícito", "'shouldve invented this sooner' es un elogio indirecto."),
    1693: ("Sentimiento implícito", "Recomendar buscar la película es positivo, pero no usa adjetivos de opinión."),
    1539: ("Negación", "'wouldnt see ... for free' niega una acción; además falta el apóstrofo."),
    1436: ("Negación", "'wasn't terrible' es positivo; el modelo pesa 'terrible' sin la negación."),
    2676: ("Negación", "'can't wait' es entusiasmo; 'can't' aparece sobre todo en reseñas negativas."),
    799: ("Negación", "'wasn't great' es negativo; el modelo pesa 'great'."),
    2594: ("Contraste de opiniones", "Mezcla 'nice' con 'NO flavor'; gana la palabra positiva."),
    1702: ("Contraste de opiniones", "Empieza con 'But ... still', que conecta con una opinión previa no incluida."),
    1490: ("Contraste de opiniones", "'few get to see, but those who do will remember it' combina tonos."),
    1899: ("Contraste de opiniones", "'brutal violence' junto a 'touching' y 'hilarious' en la misma oración."),
    2358: ("Vocabulario poco frecuente", "'fish' aparece más en reseñas negativas (4 de 6) y 'best' no alcanza a compensar."),
    2295: ("Vocabulario poco frecuente", "'fresh fish, prepared with care': palabras positivas poco vistas en entrenamiento."),
    1408: ("Vocabulario poco frecuente", "'unique' aparece solo 2 veces en el corpus; 'though' sugiere contraste."),
    1896: ("Ironía", "Recomendar la película 'para insomnes' es una crítica irónica."),
    973: ("Texto insuficiente", "Dos palabras; 'lousy' aparece solo 3 veces en todo el corpus."),
    2592: ("Lenguaje informal", "'OMG' y una comparación exagerada que el vocabulario TF-IDF no capta."),
}

PALETTE = {"fill": "#A593D3", "edge": "#6B58A6", "ink": "#0B0B0B", "muted": "#52514E", "grid": "#E1E0D9"}


def main() -> None:
    errors = pd.read_csv(REPORTS / "error_analysis.csv")
    sample = errors.sample(20, random_state=SAMPLE_SEED).copy()
    missing = set(sample["row"]) - set(REVIEW)
    if missing:
        raise ValueError(f"Filas sin revisar: {sorted(missing)}")
    sample["category"] = sample["row"].map(lambda r: REVIEW[r][0])
    sample["note"] = sample["row"].map(lambda r: REVIEW[r][1])
    sample = sample.sort_values(["category", "row"])
    sample.to_csv(REPORTS / "error_analysis_categorized.csv", index=False)

    counts = sample["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    ax.barh(counts.index, counts.values, color=PALETTE["fill"], edgecolor=PALETTE["edge"], height=0.6, zorder=3)
    for y, value in enumerate(counts.values):
        ax.text(value + 0.1, y, f"{value} ({value / len(sample):.0%})", va="center", color=PALETTE["ink"], fontsize=10)
    ax.set_xlim(0, counts.max() + 1.5)
    ax.set_xlabel("Errores en la muestra", color=PALETTE["muted"])
    ax.grid(axis="x", color=PALETTE["grid"], zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Categorías de 20 errores de Naive Bayes (muestra aleatoria, semilla 8239)", loc="left", fontsize=11, color=PALETTE["ink"])
    fig.tight_layout()
    fig.savefig(REPORTS / "error_categories.png", dpi=200)
    plt.close(fig)

    print(f"Errores totales del modelo: {len(errors)} · muestra revisada: {len(sample)}")
    print(counts.sort_values(ascending=False).to_string())
    print("\nGuardado: reports/error_analysis_categorized.csv y reports/error_categories.png")


if __name__ == "__main__":
    main()
