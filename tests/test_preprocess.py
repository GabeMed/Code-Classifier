"""Data loading and featurization. None of this needs ast2vec or torch."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import numpy as np
import pytest

import preprocess
from preprocess import (
    AST_NODE_TYPES,
    DATA_DIR,
    LABELS,
    ast2vec_features,
    ast_counts_features,
    build_dataset,
    load_labels,
    load_programs,
)


def test_programs_are_loaded_in_name_order():
    names, sources = load_programs()
    assert names == [f"exp{i}.py" for i in range(1, 8)]
    assert all(sources)


def test_labels_follow_file_names_not_directory_order(tmp_path):
    # Write the files in reverse order; the result must not depend on it.
    for name in reversed(sorted(LABELS)):
        (tmp_path / name).write_text((DATA_DIR / name).read_text())
    names, _ = load_programs(tmp_path)
    y = load_labels(names)
    assert names == sorted(LABELS)
    np.testing.assert_array_equal(y[names.index("exp7.py")], [1, 0, 0])
    np.testing.assert_array_equal(y[names.index("exp6.py")], [0, 1, 1])


def test_every_program_has_a_label_and_vice_versa():
    names, _ = load_programs()
    assert set(names) == set(LABELS)
    with pytest.raises(KeyError, match="exp99.py"):
        load_labels(["exp1.py", "exp99.py"])


def test_programs_parse_without_being_executed():
    _, sources = load_programs()
    for src in sources:
        assert isinstance(ast.parse(src), ast.Module)


def test_ast_counts_on_a_known_snippet():
    X = ast_counts_features(["x = 1"])
    counts = dict(zip(AST_NODE_TYPES, X[0], strict=True))
    nonzero = {k: v for k, v in counts.items() if v}
    assert nonzero == {k: 0.2 for k in ("Module", "Assign", "Name", "Store", "Constant")}


def test_ast_counts_are_normalised_and_deterministic():
    _, sources = load_programs()
    X = ast_counts_features(sources)
    assert X.shape == (7, len(AST_NODE_TYPES))
    assert (X >= 0).all()
    np.testing.assert_allclose(X.sum(axis=1), 1.0)
    np.testing.assert_array_equal(X, ast_counts_features(sources))
    assert len({tuple(row) for row in X}) == 7  # every program gets a distinct vector


def test_vocabulary_covers_real_code_and_skips_deprecated_nodes():
    assert "Constant" in AST_NODE_TYPES
    assert not set(AST_NODE_TYPES) & {"Num", "Str", "Index", "Param"}
    assert len(AST_NODE_TYPES) == len(set(AST_NODE_TYPES))
    for module in (json, inspect, textwrap, preprocess):
        ast_counts_features([inspect.getsource(module)])  # raises KeyError on an unknown node type


def test_ast2vec_missing_gives_setup_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="make ast2vec"):
        ast2vec_features(["x = 1"], ast2vec_dir=tmp_path / "missing")


def test_unknown_featurizer_is_rejected():
    with pytest.raises(ValueError, match="unknown featurizer"):
        build_dataset("bag-of-words")


def test_build_dataset_shapes():
    X, y, names = build_dataset("ast-counts")
    assert X.shape[0] == y.shape[0] == len(names) == 7
    assert y.shape[1] == 3
    assert set(np.unique(y)) <= {0, 1}
