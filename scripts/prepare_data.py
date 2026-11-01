"""LAB04 · Preparación del corpus Sentiment Labelled Sentences (UCI).

Lee los tres archivos originales (Amazon, IMDb y Yelp), cada línea con
"oración<TAB>etiqueta", y genera data/processed/dataset.csv con las columnas
text, label (negative/positive) y source. Los archivos de data/raw no se modifican.

Se lee línea por línea, sin interpretar comillas, porque algunas oraciones
contienen comillas sin cerrar que harían perder filas con un lector CSV. Solo
"\n" separa líneas: dos oraciones de IMDb contienen el carácter U+0085 (puntos
suspensivos de Windows-1252), que str.splitlines() trataría como salto de línea
y partiría la oración. Ese carácter se reemplaza por "...".

    uv run python scripts/prepare_data.py
"""
from __future__ import annotations

import pandas as pd

from inf8239_u02.config import ROOT

RAW = ROOT / "data" / "raw" / "sentiment_labelled_sentences"
OUTPUT = ROOT / "data" / "processed" / "dataset.csv"
FILES = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}
LABELS = {"0": "negative", "1": "positive"}


def read_source(source: str, filename: str) -> pd.DataFrame:
    matches = list(RAW.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No se encontró {filename} en {RAW}. Ejecute scripts/download_data.py")
    rows, rejected = [], 0
    content = matches[0].read_text(encoding="utf-8", errors="replace")
    for line in content.split("\n"):
        line = line.rstrip("\r").replace("\u0085", "...")
        if not line.strip():
            continue
        text, sep, label = line.rpartition("\t")
        if not sep or label.strip() not in LABELS:
            rejected += 1
            continue
        rows.append({"text": text.strip(), "label": LABELS[label.strip()], "source": source})
    print(f"{filename}: {len(rows)} oraciones leídas, {rejected} líneas descartadas")
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.concat([read_source(s, f) for s, f in FILES.items()], ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nArchivo generado: {OUTPUT.relative_to(ROOT)}")
    print(f"Filas y columnas: {df.shape}")
    print("Oraciones por fuente y etiqueta:")
    print(pd.crosstab(df["source"], df["label"], margins=True, margins_name="total"))


if __name__ == "__main__":
    main()
