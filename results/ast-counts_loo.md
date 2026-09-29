# Cross-validation: `ast-counts` features, `loo`

7 programs, 107 features, 3 binary labels.

| | Exact-match accuracy | Hamming accuracy |
|---|---:|---:|
| Random forest | 0.000 | 0.524 |
| Majority baseline | 0.286 | 0.714 |

| Label | Precision | Recall | F1 | Positives |
|---|---:|---:|---:|---:|
| t0 | 0.00 | 0.00 | 0.00 | 1 |
| t1 | 0.00 | 0.00 | 0.00 | 3 |
| t2 | 0.00 | 0.00 | 0.00 | 2 |

Seed spread (10 random-forest seeds): exact-match 0.057 ± 0.070.
Permutation test (200 label shuffles): permuted mean 0.144, p = 1.000.

| Program | True | Predicted |
|---|---|---|
| exp1.py | 000 | 010 |
| exp2.py | 000 | 010 |
| exp3.py | 001 | 010 |
| exp4.py | 010 | 000 |
| exp5.py | 010 | 000 |
| exp6.py | 011 | 000 |
| exp7.py | 100 | 010 |
