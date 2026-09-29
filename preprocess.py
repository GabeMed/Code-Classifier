"""Load the labelled programs and turn each one into a feature vector.

Two featurizers share the same interface (list of source strings -> 2-D array):

- ``ast2vec``: the 256-d encoding from the pretrained ast2vec network
  (Paaßen et al.), fetched into ``vendor/ast2vec`` by ``make ast2vec``.
- ``ast-counts``: relative frequency of each Python AST node type. Standard
  library only; used as a baseline and to test the pipeline without ast2vec.

The programs in ``data/`` are only parsed, never executed.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AST2VEC_DIR = ROOT / "vendor" / "ast2vec"

# One binary target per optimization technique (multi-label). Keyed by file
# name so the pairing does not depend on os.listdir order. The order below is
# the order the labels were originally written in (exp1 ... exp7).
LABELS = {
    "exp1.py": [0, 0, 0],
    "exp2.py": [0, 0, 0],
    "exp3.py": [0, 0, 1],
    "exp4.py": [0, 1, 0],
    "exp5.py": [0, 1, 0],
    "exp6.py": [0, 1, 1],
    "exp7.py": [1, 0, 0],
}
LABEL_NAMES = ["t0", "t1", "t2"]

FEATURIZERS = ("ast2vec", "ast-counts")


def load_programs(path: Path | str = DATA_DIR) -> tuple[list[str], list[str]]:
    """Return (file names, source code) for every .py file in ``path``, sorted by name."""
    files = sorted(Path(path).glob("*.py"))
    return [f.name for f in files], [f.read_text() for f in files]


def load_labels(names: list[str], labels: dict[str, list[int]] = LABELS) -> np.ndarray:
    missing = [n for n in names if n not in labels]
    if missing:
        raise KeyError(f"no label for: {', '.join(missing)}")
    return np.array([labels[n] for n in names])


# Deprecated aliases that ast.parse never produces.
_DEPRECATED_NODES = {
    "AugLoad", "AugStore", "Bytes", "Ellipsis", "ExtSlice", "Index",
    "NameConstant", "Num", "Param", "Str", "Suite",
}


def _concrete_node_types() -> list[str]:
    """Every concrete AST node class in the running Python, sorted.

    A fixed vocabulary keeps the feature layout independent of which programs
    are in the corpus (no fitting on held-out programs).
    """
    classes, stack = set(), [ast.AST]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub.__module__ == "ast" and sub not in classes:
                classes.add(sub)
                stack.append(sub)
    leaves = {
        c.__name__
        for c in classes
        if c.__name__ == "Constant" or not any(s.__module__ == "ast" for s in c.__subclasses__())
    }
    return sorted(leaves - _DEPRECATED_NODES)


AST_NODE_TYPES = _concrete_node_types()


def ast_counts_features(sources: list[str]) -> np.ndarray:
    """Relative frequency of each AST node type, one row per program."""
    index = {name: i for i, name in enumerate(AST_NODE_TYPES)}
    X = np.zeros((len(sources), len(AST_NODE_TYPES)))
    for row, src in enumerate(sources):
        counts = Counter(type(node).__name__ for node in ast.walk(ast.parse(src)))
        for name, count in counts.items():
            X[row, index[name]] = count
        X[row] /= X[row].sum()
    return X


def ast2vec_features(sources: list[str], ast2vec_dir: Path = AST2VEC_DIR) -> np.ndarray:
    """256-d ast2vec encoding of each program (needs ``make ast2vec`` and torch)."""
    model_path = ast2vec_dir / "ast2vec.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"ast2vec not found at {ast2vec_dir}. Run `make ast2vec` "
            "(clones https://github.com/inpefess/ast2vec at a pinned commit) and "
            "install requirements-ast2vec.txt."
        )
    # ast2vec's modules import each other by bare name (`import tree`).
    if str(ast2vec_dir) not in sys.path:
        sys.path.insert(0, str(ast2vec_dir))
    import ast2vec
    import python_ast_utils

    # Parsing to AST
    trees = [python_ast_utils.ast_to_tree(ast.parse(src)) for src in sources]

    # Loading the model
    model = ast2vec.load_model(str(model_path))

    # Generating the code vectors
    return np.array([model.encode(tree).detach().numpy() for tree in trees])


def build_dataset(features: str = "ast2vec", path: Path | str = DATA_DIR):
    """Return (X, y, names) for the programs in ``path``."""
    names, sources = load_programs(path)
    y = load_labels(names)
    if features == "ast2vec":
        X = ast2vec_features(sources)
    elif features == "ast-counts":
        X = ast_counts_features(sources)
    else:
        raise ValueError(f"unknown featurizer {features!r}; choose from {FEATURIZERS}")
    return X, y, names
