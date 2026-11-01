"""LAB04 · Descarga reproducible del corpus.

Modo local: solo informa el archivo configurado.
Modo URL: descarga DATASET_URL, imprime su SHA-256 y, si es un ZIP, lo extrae en
data/raw/<nombre>/ sin modificar los archivos originales.

    uv run python scripts/download_data.py
"""
from __future__ import annotations

import sys
import zipfile
from urllib.parse import unquote, urlparse

import requests

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import sha256


def main() -> int:
    if settings.data_source == "local":
        print(f"Modo local. Archivo configurado: {settings.dataset_path}")
        return 0
    if settings.data_source != "url" or not settings.dataset_url:
        print("Configure DATA_SOURCE=url y DATASET_URL en .env", file=sys.stderr)
        return 2

    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    name = unquote(urlparse(settings.dataset_url).path.rsplit("/", 1)[-1]).replace("+", "_") or "dataset.csv"
    output = raw / name
    response = requests.get(settings.dataset_url, timeout=60)
    response.raise_for_status()
    output.write_bytes(response.content)
    print(f"Fuente: {settings.dataset_url}")
    print(f"Guardado: {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(output)}")

    if zipfile.is_zipfile(output):
        destination = raw / output.stem
        with zipfile.ZipFile(output) as archive:
            members = [m for m in archive.namelist() if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")]
            archive.extractall(destination, members=members)
        print(f"Extraído en: {destination.relative_to(ROOT)}")
        for path in sorted(destination.rglob("*.txt")):
            print(f"  {path.relative_to(destination)} · SHA-256 {sha256(path)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
