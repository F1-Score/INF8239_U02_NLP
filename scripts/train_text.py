from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset, validate_dataframe
from inf8239_u02.modeling import build_models


def main() -> None:
    df = load_dataset().dropna(subset=[settings.text_column, settings.target_column])
    df = df.drop_duplicates(subset=[settings.text_column])
    validate_dataframe(df, settings.text_column, settings.target_column)
    counts = df[settings.target_column].value_counts()
    stratify = df[settings.target_column] if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        df[settings.text_column], df[settings.target_column], test_size=0.25,
        random_state=settings.random_state, stratify=stratify,
    )
    models = build_models(settings.random_state)
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[name] = pred
        score = f1_score(y_test, pred, average="macro")
        labels = sorted(y_test.unique())
        recalls = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
        rows.append({
            "model": name,
            "f1_macro": score,
            "accuracy": accuracy_score(y_test, pred),
            **{f"recall_{label}": value for label, value in zip(labels, recalls)},
        })
        print(f"\n{name} · F1 macro={score:.3f}")
        print(classification_report(y_test, pred, zero_division=0))
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    table.to_csv(reports / "text_metrics.csv", index=False)
    # Se guarda el pipeline con mayor F1 macro en la prueba (el proyecto base fijaba "logistic").
    best = table.iloc[0]["model"]
    print(f"\nModelo seleccionado por F1 macro: {best}")
    selected = models[best]
    selected_pred = predictions[best]
    pastel = LinearSegmentedColormap.from_list("lavanda", ["#F7F5FC", "#A593D3", "#6B58A6"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(y_test, selected_pred, cmap=pastel, colorbar=False, ax=ax)
    ax.set_title(f"Matriz de confusión · {best} (n = {len(y_test)})", fontsize=11, loc="left")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(reports / "confusion_text.png", dpi=200)
    plt.close(fig)
    source = df.loc[x_test.index, "source"] if "source" in df.columns else ""
    errors = pd.DataFrame({"text": x_test, "source": source, "real": y_test, "predicted": selected_pred})
    errors = errors[errors["real"] != errors["predicted"]].copy()
    errors["category"] = "REVISAR"
    errors.to_csv(reports / "error_analysis.csv", index_label="row")
    joblib.dump(selected, models_dir / "text_model.joblib")
    print("\nArtefactos guardados en reports/ y models/")


if __name__ == "__main__":
    main()
