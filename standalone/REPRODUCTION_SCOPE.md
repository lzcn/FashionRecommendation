# End-to-end reproduction scope

## Tensor decomposition

The four in-scope tag data sets are represented as binary user-item-tag tensors.
The paper says that their rank-64 CP factors were learned with BPTF and a BPR
pairwise-ranking objective. The released `mex/train/factorization.cpp` instead
maximizes the sigmoid of each positive-vs-negative score difference and enumerates
all negative tags for every user-item post.

The standalone trainer therefore exposes both interpretations and both update
schedules:

- `--loss bpr`: standard log-sigmoid BPR objective;
- `--loss legacy-auc`: sigmoid/AUC surrogate matching the released MEX gradient;
- `--negative-mode sampled`: scalable, deterministic negative sampling;
- `--negative-mode all`: enumerate negatives with immediate pairwise SGD;
- `--optimizer legacy-batch`: aggregate and normalize all positive-negative
  gradients per user-item post, matching the released MEX update schedule.

All runs record seed, rank, learning rate, regularization, epoch metrics, and the
resulting factor files. Real-data production runs use rank 64 as stated in the paper.

## Search methods

| Method | Role | Final-paper status |
|---|---|---|
| Exhaustive search | exact ground truth and timing baseline | included |
| Diamond sampling | prior MAD baseline for the matrix special case | included |
| Core^1 | proposed method / ablation | included |
| Core^2 | proposed method / ablation | included |
| Core^3 | proposed method / ablation | included |
| Wedge sampling | older repository baseline | optional; not a final-paper baseline |

The deleted Git-history drivers confirm sample counts from 10^3 to 10^8,
top-t values in {1, 10, 100, 1000}, maximum prefilter budget t'=s, and ten
repetitions for the main curves. Smaller smoke profiles are used before launching
the full experiment matrix.

DBLP and its CP-ALS decomposition are excluded at the user's request.
