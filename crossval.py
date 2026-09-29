"""Cross-validated evaluation of the random-forest code classifier.

Usage:
    python crossval.py --features ast2vec      # ast2vec vectors (needs `make ast2vec`)
    python crossval.py --features ast-counts   # AST node-count baseline, standard library only

Protocol:
- Leave-one-out over the programs by default (``--cv kfold3`` reproduces the
  original 3-fold split). Every program is predicted by a model that never saw it.
- StandardScaler sits inside the pipeline, so it is refit on each training fold.
- Baseline: per-label majority class (DummyClassifier), same folds.
- Robustness: the random-forest seed is varied, and a label-permutation test
  asks whether the exact-match accuracy is distinguishable from chance.

Writes ``results/<features>_<cv>.json`` and ``.md`` plus a confusion-matrix figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURIZERS, LABEL_NAMES, build_dataset

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 42


def make_classifier(random_state: int = RANDOM_STATE) -> Pipeline:
    # Normalization + random forest, as in the original script. The scaler is
    # inside the pipeline so cross-validation refits it on each training fold.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ]
    )


def make_cv(name: str):
    if name == "loo":
        return LeaveOneOut()
    if name == "kfold3":
        return KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    raise ValueError(f"unknown cv {name!r}")


def combo(row) -> str:
    """Label vector [0, 1, 1] -> '011', so each combination is one class."""
    return "".join(str(int(v)) for v in row)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    true_c = [combo(r) for r in y_true]
    pred_c = [combo(r) for r in y_pred]
    combos = sorted(set(true_c) | set(pred_c))
    return {
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_accuracy": float(1 - hamming_loss(y_true, y_pred)),
        "per_label": [
            {
                "label": name,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
                "confusion": cm.tolist(),
            }
            for name, p, r, f, s, cm in zip(
                LABEL_NAMES, precision, recall, f1, support, multilabel_confusion_matrix(y_true, y_pred), strict=True
            )
        ],
        "combinations": combos,
        "combination_confusion": confusion_matrix(true_c, pred_c, labels=combos).tolist(),
    }


def evaluate(X: np.ndarray, y: np.ndarray, cv, estimator=None) -> tuple[dict, np.ndarray]:
    estimator = make_classifier() if estimator is None else estimator
    y_pred = cross_val_predict(estimator, X, y, cv=cv)
    return score_predictions(y, y_pred), y_pred


def seed_spread(X, y, cv, seeds: tuple[int, ...] = tuple(range(10))) -> dict:
    scores = [accuracy_score(y, cross_val_predict(make_classifier(s), X, y, cv=cv)) for s in seeds]
    return {"seeds": list(seeds), "exact_match": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}


def permutation_pvalue(X, y, cv, n_permutations: int) -> dict:
    score, perm_scores, pvalue = permutation_test_score(
        make_classifier(),
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_permutations=n_permutations,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return {
        "n_permutations": n_permutations,
        "score": float(score),
        "permuted_mean": float(np.mean(perm_scores)),
        "p_value": float(pvalue),
    }


def plot_confusion(result: dict, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    for ax, lab in zip(axes[:3], result["per_label"], strict=True):
        ConfusionMatrixDisplay(np.array(lab["confusion"]), display_labels=[0, 1]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"label {lab['label']}")
    ConfusionMatrixDisplay(
        np.array(result["combination_confusion"]), display_labels=result["combinations"]
    ).plot(ax=axes[3], colorbar=False, cmap="Blues")
    axes[3].set_title("label combination")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def to_markdown(report: dict) -> str:
    m, b = report["model"], report["baseline"]
    lines = [
        f"# Cross-validation: `{report['features']}` features, `{report['cv']}`",
        "",
        f"{report['n_programs']} programs, {report['n_features']} features, {len(LABEL_NAMES)} binary labels.",
        "",
        "| | Exact-match accuracy | Hamming accuracy |",
        "|---|---:|---:|",
        f"| Random forest | {m['exact_match_accuracy']:.3f} | {m['hamming_accuracy']:.3f} |",
        f"| Majority baseline | {b['exact_match_accuracy']:.3f} | {b['hamming_accuracy']:.3f} |",
        "",
        "| Label | Precision | Recall | F1 | Positives |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {lab['label']} | {lab['precision']:.2f} | {lab['recall']:.2f} | {lab['f1']:.2f} | {lab['support']} |"
        for lab in m["per_label"]
    ]
    s, p = report["seed_spread"], report["permutation_test"]
    lines += [
        "",
        f"Seed spread ({len(s['seeds'])} random-forest seeds): exact-match {s['mean']:.3f} ± {s['std']:.3f}.",
        f"Permutation test ({p['n_permutations']} label shuffles): permuted mean {p['permuted_mean']:.3f}, "
        f"p = {p['p_value']:.3f}.",
        "",
        "| Program | True | Predicted |",
        "|---|---|---|",
    ]
    rows = zip(report["programs"], report["true"], report["predicted"], strict=True)
    lines += [f"| {n} | {t} | {pr} |" for n, t, pr in rows]
    return "\n".join(lines) + "\n"


def run(features: str, cv_name: str, n_permutations: int, out_dir: Path) -> dict:
    X, y, names = build_dataset(features)
    cv = make_cv(cv_name)

    model, y_pred = evaluate(X, y, cv)
    baseline, _ = evaluate(X, y, cv, DummyClassifier(strategy="most_frequent"))
    report = {
        "features": features,
        "cv": cv_name,
        "n_programs": len(names),
        "n_features": int(X.shape[1]),
        "model": model,
        "baseline": baseline,
        "seed_spread": seed_spread(X, y, cv),
        "permutation_test": permutation_pvalue(X, y, cv, n_permutations),
        "programs": names,
        "true": [combo(r) for r in y],
        "predicted": [combo(r) for r in y_pred],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{features}_{cv_name}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / f"{stem}.md").write_text(to_markdown(report))
    plot_confusion(model, f"Out-of-fold predictions, {features} features, {cv_name}", out_dir / f"{stem}_confusion.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=FEATURIZERS, default="ast2vec")
    parser.add_argument("--cv", choices=["loo", "kfold3"], default="loo")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    report = run(args.features, args.cv, args.permutations, args.out)
    print(to_markdown(report))


if __name__ == "__main__":
    main()
