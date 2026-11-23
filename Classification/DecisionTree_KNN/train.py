"""Raw-pixel features -> Decision Tree / KNN (library + NumPy, with and without PCA) + PCA analysis + report figures.

Run: uv run Classification/DecisionTree_KNN/train.py   (outputs go to Classification/results/)
"""
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from scratch import PCA, KNN, DecisionTree

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "trashnet" / "dataset-resized"
RESULTS = HERE.parent / "results"
OUT = RESULTS / "raw"  # raw pixels as features
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)

IMPLS = {  # same hyperparameter names for both, so one grid serves both
    "library": {"DT": DecisionTreeClassifier(random_state=SEED), "KNN": KNeighborsClassifier(), "PCA": SkPCA(random_state=SEED)},
    "numpy": {"DT": DecisionTree(), "KNN": KNN(), "PCA": PCA()},
}
GRIDS = {
    "DT": {"clf__max_depth": [3, 5, 8, 12, None], "clf__min_samples_leaf": [1, 3, 5]},
    "KNN": {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31], "clf__weights": ["uniform", "distance"]},
}
PCA_VARIANCE = [0.80, 0.90, 0.95]  # tuned too, for the "+PCA" variants


def raw_features(path):
    # no feature engineering: 32x24 RGB pixels scaled to [0, 1] -> 2304 features
    return np.asarray(Image.open(path).convert("RGB").resize((32, 24)), dtype=float).ravel() / 255


def macro_roc(y, proba):
    """Macro-average one-vs-rest ROC curve on a common FPR grid."""
    grid = np.linspace(0, 1, 201)
    tprs = [np.interp(grid, *roc_curve(y == k, proba[:, k])[:2]) for k in range(proba.shape[1])]
    return grid, np.mean(tprs, 0)


def md_table(df):
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def tune(X_tr, y_tr, X_te, y_te, classes):
    """GridSearchCV (5-fold) for every algo x with/without PCA x library/numpy; metrics on the test set."""
    res, reports = {}, []
    for impl in IMPLS:
        for algo in ("DT", "KNN"):
            for use_pca in (False, True):
                steps = ([("pca", IMPLS[impl]["PCA"])] if use_pca else []) + [("clf", IMPLS[impl][algo])]
                grid = dict(GRIDS[algo], **({"pca__n_components": PCA_VARIANCE} if use_pca else {}))
                name = f"{algo}{' + PCA' if use_pca else ''} ({impl})"
                t0 = time.time()
                gs = GridSearchCV(Pipeline(steps), grid, cv=CV, n_jobs=N_JOBS).fit(X_tr, y_tr)
                proba = gs.predict_proba(X_te)
                pred = proba.argmax(1)
                p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
                res[name] = dict(
                    algo=algo, impl=impl, pca=use_pca, proba=proba, pred=pred,
                    row={"model": name, "best_params": json.dumps(gs.best_params_, default=str),
                         "cv_acc": gs.best_score_, "train_acc": gs.score(X_tr, y_tr), "test_acc": (pred == y_te).mean(),
                         "precision": p, "recall": r, "f1": f,
                         "roc_auc": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
                         "fit_s": gs.refit_time_})
                reports.append(f"===== {name} =====\n"
                               f"{classification_report(y_te, pred, target_names=classes, digits=3, zero_division=0)}")
                print(f"{name:<22} cv {gs.best_score_:.3f} test {res[name]['row']['test_acc']:.3f} "
                      f"auc {res[name]['row']['roc_auc']:.3f} | {gs.best_params_} | {time.time() - t0:.0f}s", flush=True)
    return res, reports


def pca_analysis(X, y, classes):
    """Assignment 3(b): PCA from the covariance matrix, eigenvalues, explained variance, 2/3 PCs, plots."""
    pca = PCA(n_components=X.shape[1]).fit(X)
    ratio, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    ref = SkPCA().fit(X)
    print("PCA check vs sklearn: max |ratio diff| =", np.abs(ratio - ref.explained_variance_ratio_).max())

    pd.DataFrame({"component": np.arange(1, 21), "eigenvalue": pca.eigenvalues_[:20],
                  "explained_variance_ratio": ratio[:20], "cumulative": cum[:20]}).to_csv(OUT / "pca_eigenvalues.csv", index=False)
    np.savetxt(OUT / "pca_eigenvectors_top10.csv", pca.components_[:, :10], delimiter=",",
               header=",".join(f"PC{i}" for i in range(1, 11)), comments="")  # column j = eigenvector of PC j
    need = {f"{int(v * 100)}%": int(np.searchsorted(cum, v) + 1) for v in (0.8, 0.9, 0.95, 0.99)}
    print("components needed for variance:", need)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.arange(1, 51), pca.eigenvalues_[:50], "o-", ms=3)
    ax[0].set(title="Eigenvalues (first 50)", xlabel="component", ylabel="eigenvalue")
    ax[1].plot(np.arange(1, len(cum) + 1), cum)
    for v in (0.9, 0.95):
        ax[1].axhline(v, ls="--", c="gray", lw=0.8)
    ax[1].set(title="Cumulative explained variance", xlabel="number of components", ylabel="cumulative ratio")
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance.png", dpi=130)
    plt.close(fig)

    Z2 = PCA(2).fit_transform(X)
    Z3 = PCA(3).fit_transform(X)
    np.savez(OUT / "pca_reduced.npz", Z2=Z2, Z3=Z3, y=y)  # the data reduced to 2 and 3 components
    print(f"2 PCs keep {ratio[:2].sum():.1%} of variance, 3 PCs keep {ratio[:3].sum():.1%}; shapes {Z2.shape}, {Z3.shape}")
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, c in enumerate(classes):
        ax.scatter(*Z2[y == k].T, s=8, label=c, alpha=0.7)
    ax.set(title=f"First two principal components ({ratio[:2].sum():.1%} variance)", xlabel="PC1", ylabel="PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pca_2d.png", dpi=130)
    plt.close(fig)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    for k, c in enumerate(classes):
        ax.scatter(*Z3[y == k].T, s=6, label=c, alpha=0.7)
    ax.set(title=f"First three principal components ({ratio[:3].sum():.1%} variance)", xlabel="PC1", ylabel="PC2", zlabel="PC3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pca_3d.png", dpi=130)
    plt.close(fig)
    return need


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted((DATA / c).glob("*.jpg"))]
    X = np.array([raw_features(f) for f, _ in files])
    y = np.array([i for _, i in files])
    print("features:", X.shape, flush=True)

    # 80/20 stratified; same seed/ordering as the ResNet split, so the test set is identical
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # ---------- tune (5-fold CV on the train set) and evaluate on the test set; cached (delete the pkl to redo) ----------
    cache = OUT / "classical_cache.pkl"
    if cache.exists():
        res, reports = pickle.loads(cache.read_bytes())
    else:
        res, reports = tune(X_tr, y_tr, X_te, y_te, classes)
        cache.write_bytes(pickle.dumps((res, reports)))

    # ---------- CNN results (written by Classification/ResNet/train.py) ----------
    cnn_file, hist_file = RESULTS / "cnn_test.npz", RESULTS / "cnn_history.csv"
    if cnn_file.exists() and hist_file.exists():
        d = np.load(cnn_file)
        assert (d["y"] == y_te).all(), "CNN test set differs from the classical test set"
        proba = d["proba"] / d["proba"].sum(1, keepdims=True)
        pred = proba.argmax(1)
        hist = pd.read_csv(hist_file)
        b = hist.loc[hist.val_acc.idxmax()]
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
        name = "ResNet-50 CNN (scratch)"
        res[name] = dict(algo="CNN", impl="numpy", pca=False, proba=proba, pred=pred, row={
            "model": name, "best_params": "SGD lr=0.02, 100 epochs", "cv_acc": b.val_acc, "train_acc": b.train_acc,
            "test_acc": (pred == y_te).mean(), "precision": p, "recall": r, "f1": f,
            "roc_auc": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"), "fit_s": float("nan")})
        reports.append(f"===== {name} =====\n"
                       f"{classification_report(y_te, pred, target_names=classes, digits=3, zero_division=0)}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, col, t in ((ax[0], "loss", "Loss"), (ax[1], "acc", "Accuracy")):
            a.plot(hist.epoch, hist[f"train_{col}"], label="train (augmented)")
            a.plot(hist.epoch, hist[f"val_{col}"], label="validation")
            a.set(title=f"CNN {t}", xlabel="epoch")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "cnn_curves.png", dpi=130)
        plt.close(fig)
    else:
        print("CNN results not found yet (run Classification/ResNet/train.py), skipping CNN rows")

    # ---------- tables ----------
    df = pd.DataFrame([v["row"] for v in res.values()])
    df.to_csv(OUT / "metrics.csv", index=False)
    notes = ["", "cv_acc = 5-fold CV accuracy on the train set (CNN: held-out validation accuracy); "
             "train_acc = accuracy on the training data (CNN: augmented batches at the best-val epoch).", ""]
    for algo in ("DT", "KNN"):
        for suffix in ("", " + PCA"):
            a, b = res[f"{algo}{suffix} (library)"], res[f"{algo}{suffix} (numpy)"]
            notes.append(f"- {algo}{suffix}: library vs NumPy test predictions agree on {(a['pred'] == b['pred']).mean():.1%}")
    best = df.loc[df.roc_auc.idxmax()]
    notes += ["", f"Best model by macro ROC-AUC: **{best.model}** (AUC {best.roc_auc:.3f})"]
    (OUT / "metrics.md").write_text(md_table(df) + "\n" + "\n".join(notes) + "\n", encoding="utf-8")
    (OUT / "reports.txt").write_text("\n".join(reports), encoding="utf-8")

    # ---------- confusion matrices ----------
    n = len(res)
    cols = 3
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(5 * cols, 4.6 * -(-n // cols)))
    for ax, (name, v) in zip(axes.ravel(), res.items()):
        ConfusionMatrixDisplay.from_predictions(y_te, v["pred"], display_labels=classes, ax=ax, cmap="Blues",
                                                xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name}\nacc {v['row']['test_acc']:.3f}", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    # ---------- ROC (macro-average, one-vs-rest) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, impl, title in ((axes[0], "library", "Library implementations"), (axes[1], "numpy", "NumPy (from scratch) implementations")):
        for name, v in res.items():
            if v["algo"] == "CNN" or v["impl"] == impl:
                ax.plot(*macro_roc(y_te, v["proba"]), label=f"{name}  AUC={v['row']['roc_auc']:.3f}",
                        ls="--" if v["pca"] else "-", lw=2.5 if v["algo"] == "CNN" else 1.5)
        ax.plot([0, 1], [0, 1], ":", c="gray")
        ax.set(title=f"{title}: macro-average ROC", xlabel="False positive rate")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=130)
    plt.close(fig)

    # ---------- overfitting vs model complexity (train vs CV accuracy) ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    depths = list(range(1, 26))
    ks = list(range(1, 42, 2))
    for a, est, param, vals, title, xlab in (
            (ax[0], DecisionTreeClassifier(random_state=SEED), "max_depth", depths, "Decision Tree", "max_depth (complexity ->)"),
            (ax[1], KNeighborsClassifier(), "n_neighbors", ks, "KNN", "k (complexity <-)")):
        tr, cv = validation_curve(est, X_tr, y_tr, param_name=param, param_range=vals, cv=CV, n_jobs=N_JOBS)
        a.plot(vals, tr.mean(1), "o-", ms=3, label="train")
        a.plot(vals, cv.mean(1), "o-", ms=3, label="5-fold CV")
        a.set(title=f"{title}: overfitting vs complexity", xlabel=xlab, ylabel="accuracy")
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "overfitting_dt_knn.png", dpi=130)
    plt.close(fig)

    # ---------- PCA analysis (assignment 3b) ----------
    pca_analysis(X, y, classes)
    print("\n" + md_table(df.drop(columns="best_params")))
    print("\n".join(notes))


if __name__ == "__main__":  # required on Windows: joblib workers re-import this file
    main()
