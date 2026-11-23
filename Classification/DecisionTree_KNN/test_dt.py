"""Test Decision Tree (library + NumPy, with and without PCA) on the held-out test split.

Models aren't saved by train.py, so each is refit on the train split with the best CV params
from results/raw/metrics.csv, then scored on the test split (same 80/20 split as train.py).
Run: python test_dt.py   (from this folder)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train import DATA, IMPLS, OUT, SEED, raw_features

ALGO = "DT"

if __name__ == "__main__":
    classes = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted((DATA / c).glob("*.jpg"))]
    X = np.array([raw_features(f) for f, _ in files])
    y = np.array([i for _, i in files])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    best = pd.read_csv(OUT / "metrics.csv").set_index("model")["best_params"]
    rows = []
    for impl in IMPLS:
        for use_pca in (False, True):
            name = f"{ALGO}{' + PCA' if use_pca else ''} ({impl})"
            steps = ([("pca", IMPLS[impl]["PCA"])] if use_pca else []) + [("clf", IMPLS[impl][ALGO])]
            model = Pipeline(steps).set_params(**json.loads(best[name])).fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)
            pred = proba.argmax(1)
            rows.append({"model": name, "test_acc": accuracy_score(y_te, pred),
                         "macro_f1": f1_score(y_te, pred, average="macro"),
                         "roc_auc": roc_auc_score(y_te, proba, multi_class="ovr", average="macro")})
            print(f"===== {name} =====")
            print(classification_report(y_te, pred, target_names=classes, digits=3, zero_division=0))
            print(confusion_matrix(y_te, pred), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT.parent / "test_metrics_dt.csv", index=False)
    print("\n" + df.to_string(index=False, float_format="%.3f"))
