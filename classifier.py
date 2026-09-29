"""Single train/test split demo. Use crossval.py for the reported numbers.

Usage:
    python classifier.py --features ast2vec [--show]
"""

from __future__ import annotations

import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

from crossval import RANDOM_STATE, RESULTS_DIR, combo, make_classifier
from preprocess import FEATURIZERS, LABEL_NAMES, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=FEATURIZERS, default="ast2vec")
    parser.add_argument("--show", action="store_true", help="open the confusion-matrix window")
    args = parser.parse_args()

    X, y, names = build_dataset(args.features)

    # Train/test split (the scaler is fit on the training part only, inside the pipeline)
    X_train, X_test, y_train, y_test, _, names_test = train_test_split(
        X, y, names, test_size=0.3, random_state=RANDOM_STATE
    )

    # Normalization + RandomForest classifier
    clf = make_classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Exact-match accuracy: all three labels of a program must be right
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test programs: {len(y_test)}  exact-match accuracy: {accuracy * 100:.2f}%")

    # Predictions vs. true values
    for name, pred, true in zip(names_test, y_pred, y_test, strict=True):
        print(f"{name}  prediction: {combo(pred)}  true: {combo(true)}")

    # One 2x2 confusion matrix per label. (Taking argmax over the label vector
    # would merge "no technique" [0,0,0] with "technique 0" [1,0,0].)
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(10, 3.4))
    for ax, name, cm in zip(axes, LABEL_NAMES, multilabel_confusion_matrix(y_test, y_pred), strict=True):
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"label {name}")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.features}_split_confusion.png"
    fig.savefig(out, dpi=90)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
