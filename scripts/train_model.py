"""Entrena el Random Forest del ejemplo y guarda artefactos reproducibles."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "examples"


def train() -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, max_depth=8, random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 5),
        "precision_macro": round(float(precision_score(y_test, pred, average="macro")), 5),
        "recall_macro": round(float(recall_score(y_test, pred, average="macro")), 5),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 5),
        "matriz_confusion_0_maligno_1_benigno": confusion_matrix(y_test, pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    probabilities = model.predict_proba(X_test)
    def select_example(target_class: int) -> int:
        # Ejemplo correctamente clasificado con probabilidad no extrema,
        # más útil para interpretar las respuestas de una API demostrativa.
        options = [i for i in range(len(y_test)) if y_test[i] == target_class and pred[i] == target_class]
        return int(min(options, key=lambda i: abs(float(max(probabilities[i])) - 0.87)))

    chosen = select_example(0)  # Caso de etiqueta 0: maligno
    other = select_example(1)  # Caso de etiqueta 1: benigno
    example = {"features": [float(v) for v in X_test[chosen].tolist()]}
    example_other = {"features": [float(v) for v in X_test[other].tolist()]}
    metadata = {
        "model_name": "RandomForest_breast_cancer_demo",
        "version": "1.0.0",
        "target_labels": {"0": "maligno", "1": "benigno"},
        "feature_names": list(map(str, dataset.feature_names)),
        "n_features": X.shape[1],
        "dataset": "Breast Cancer Wisconsin Diagnostic / scikit-learn",
        "random_state": 42,
        "test_size": 0.20,
        "hyperparameters": {"n_estimators": 250, "max_depth": 8, "random_state": 42, "n_jobs": 1},
        "example": example,
        "example_expected_class_id": int(y_test[chosen]),
        "sklearn_version": sklearn.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Demostración académica: no utilizar para decisiones médicas.",
    }
    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (ARTIFACTS / "training_report.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXAMPLES / "sample_request.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    (EXAMPLES / "batch_request.json").write_text(json.dumps({"instances": [example, example_other]}, indent=2), encoding="utf-8")
    print("Modelo entrenado y exportado:", ARTIFACTS / "model.joblib")
    print("Métricas:", json.dumps(scores, ensure_ascii=False))
    return scores


if __name__ == "__main__":
    train()
