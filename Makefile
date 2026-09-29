# Entry points for the code classifier. Run `make help` for the list.

PY ?= .venv/bin/python
FEATURES ?= ast2vec
AST2VEC_REPO = https://github.com/inpefess/ast2vec
AST2VEC_COMMIT = f144dcdb6a094caf3a58bc4940c88348a5264e91

.PHONY: help setup ast2vec cv split baseline lint test

help:
	@echo "make setup      create .venv (Python 3.11) with the pinned environment"
	@echo "make ast2vec    fetch ast2vec (pinned commit) into vendor/ and install torch"
	@echo "make cv         leave-one-out and 3-fold CV -> results/ (FEATURES=ast2vec|ast-counts, default ast2vec)"
	@echo "make split      single 70/30 split demo"
	@echo "make baseline   CV with the AST node-count baseline (no ast2vec needed)"
	@echo "make test       lint + tests (no ast2vec needed)"

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.lock

ast2vec:
	@test -d vendor/ast2vec || git clone $(AST2VEC_REPO) vendor/ast2vec
	git -C vendor/ast2vec checkout --quiet $(AST2VEC_COMMIT)
	uv pip install --python $(PY) -r requirements-ast2vec.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

cv:
	$(PY) crossval.py --features $(FEATURES) --cv loo
	$(PY) crossval.py --features $(FEATURES) --cv kfold3

split:
	$(PY) classifier.py --features $(FEATURES)

baseline:
	$(MAKE) cv FEATURES=ast-counts

lint:
	$(PY) -m ruff check .

test: lint
	$(PY) -m pytest
