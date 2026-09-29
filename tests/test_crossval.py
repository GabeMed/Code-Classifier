"""Evaluation harness: metrics, leakage-safe pipeline, baseline, end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

import crossval
from crossval import combo, evaluate, make_classifier, make_cv, run, score_predictions
from preprocess import build_dataset

RESULTS = Path(crossval.__file__).resolve().parent / "results"


def test_combo_keeps_empty_and_single_technique_apart():
    # The old confusion matrix used argmax, which maps both of these to 0.
    assert combo([0, 0, 0]) == "000"
    assert combo([1, 0, 0]) == "100"
    assert combo(np.array([0, 1, 1])) == "011"


def test_score_predictions_by_hand():
    y_true = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1]])
    s = score_predictions(y_true, y_pred)

    assert s["exact_match_accuracy"] == 0.5
    assert s["hamming_accuracy"] == pytest.approx(10 / 12)
    t0, t1, t2 = s["per_label"]
    assert (t0["support"], t0["recall"]) == (1, 0.0)
    assert (t1["precision"], t1["recall"]) == (1.0, 1.0)
    assert (t2["precision"], t2["recall"]) == (0.5, 1.0)
    assert s["combinations"] == ["000", "010", "011", "100"]
    cm = np.array(s["combination_confusion"])
    assert cm.sum() == 4 and np.trace(cm) == 2
    assert cm[s["combinations"].index("100"), s["combinations"].index("000")] == 1


def test_scaler_is_part_of_the_cv_pipeline():
    pipe = make_classifier()
    assert [name for name, _ in pipe.steps] == ["scaler", "clf"]
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert pipe.named_steps["clf"].random_state == crossval.RANDOM_STATE


def test_harness_recovers_a_clean_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(40, 3))
    X = np.hstack([y + rng.normal(scale=0.05, size=y.shape), rng.normal(size=(40, 5))])
    scores, y_pred = evaluate(X, y, make_cv("kfold3"))
    assert y_pred.shape == y.shape
    assert scores["exact_match_accuracy"] >= 0.95


def test_majority_baseline_on_the_real_labels():
    X, y, _ = build_dataset("ast-counts")
    scores, y_pred = evaluate(X, y, LeaveOneOut(), DummyClassifier(strategy="most_frequent"))
    assert (y_pred == 0).all()
    assert scores["exact_match_accuracy"] == pytest.approx(2 / 7)
    assert scores["hamming_accuracy"] == pytest.approx(15 / 21)


def test_run_writes_a_consistent_report(tmp_path):
    report = run("ast-counts", "loo", n_permutations=5, out_dir=tmp_path)
    for suffix in (".json", ".md", "_confusion.png"):
        assert (tmp_path / f"ast-counts_loo{suffix}").exists()

    saved = json.loads((tmp_path / "ast-counts_loo.json").read_text())
    assert saved["n_programs"] == 7
    assert saved["true"] == ["000", "000", "001", "010", "010", "011", "100"]
    assert len(saved["predicted"]) == 7
    exact = np.mean([t == p for t, p in zip(saved["true"], saved["predicted"], strict=True)])
    assert exact == pytest.approx(report["model"]["exact_match_accuracy"])
    assert 0 < saved["permutation_test"]["p_value"] <= 1


@pytest.mark.parametrize("cv_name", ["loo", "kfold3"])
def test_committed_results_are_reproducible(tmp_path, cv_name):
    committed = json.loads((RESULTS / f"ast-counts_{cv_name}.json").read_text())
    fresh = run("ast-counts", cv_name, n_permutations=5, out_dir=tmp_path)
    for key in ("model", "baseline", "seed_spread", "predicted"):
        assert fresh[key] == committed[key], key
