# Cross-validation: `ast-counts` features, `kfold3`

7 programs, 107 features, 3 binary labels.

| | Exact-match accuracy | Hamming accuracy |
|---|---:|---:|
| Random forest | 0.143 | 0.714 |
| Majority baseline | 0.286 | 0.714 |

| Label | Precision | Recall | F1 | Positives |
|---|---:|---:|---:|---:|
| t0 | 0.00 | 0.00 | 0.00 | 1 |
| t1 | 0.50 | 0.33 | 0.40 | 3 |
| t2 | 0.00 | 0.00 | 0.00 | 2 |

Seed spread (10 random-forest seeds): exact-match 0.143 ± 0.090.
Permutation test (200 label shuffles): permuted mean 0.156, p = 0.741.

| Program | True | Predicted |
|---|---|---|
| exp1.py | 000 | 010 |
| exp2.py | 000 | 000 |
| exp3.py | 001 | 000 |
| exp4.py | 010 | 000 |
| exp5.py | 010 | 000 |
| exp6.py | 011 | 010 |
| exp7.py | 100 | 000 |
