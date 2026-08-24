# Standalone C++ reproduction

This directory contains a MATLAB-independent C++17 implementation of Algorithm 1
from *Sampling for Approximate Maximum Search in Factorized Tensor* (IJCAI 2017).
It implements exhaustive top-*t* search and Core^1, Core^2, and Core^3 sampling,
including compound-core randomized rounding, signed sampling scores, optional
prefiltering, recall measurement, fixed seeds, and phase timings.

Build and test:

```sh
make -C standalone
make -C standalone test
```

Convert a raw tag-assignment file to an iterative 5-core with dense zero-based
IDs (use `--format ml10m` or `--format ml20m` for MovieLens, and the default
`tsv` for HetRec):

```sh
./standalone/prepare_data --input RAW_FILE --output triples.csv --min-count 5
```

For free-text tags, `--lowercase-tags` can be used to merge case variants.
`--legacy-mex-core` reproduces the released MEX trainer's historical behavior:
raw rows are counted before duplicate triples are collapsed.

Train rank-64 CP factors with the scalable pairwise trainer:

```sh
./standalone/factorize \
  --input data/processed/ml_s/triples.csv \
  --output-dir results/factors/ml_s \
  --rank 64 --epochs 20 --negatives 1 --loss bpr --seed 2017
```

Use `--optimizer legacy-batch` to reproduce the released MEX trainer's aggregate
all-negative update. It is intentionally not the default because its cost is
proportional to every positive-negative tag pair; one ML-S epoch evaluates about
33 million pairs. `--loss legacy-auc --negative-mode all` instead performs
immediate pairwise SGD and is retained only as a diagnostic variant.

The RTF paper cited by the manuscript initializes factors from `N(0, 0.1)` and
uses hundreds of iterations. Reproduce that initialization explicitly with
`--init-distribution normal --init-scale 0.1`; the released MEX currently calls
`RANDU`, whose implementation ignores its nominal mean and standard-deviation
arguments.

Run the matrix-special-case baselines used for the paper's Diamond comparison:

```sh
./standalone/matrix_benchmark \
  --factor-dir results/factors/ml_s_bpr --users 10 \
  --methods diamond,wedge,core1,core2,core3 \
  --samples 10000,100000 --top 1,10,100,1000 \
  --budget-ratio 0.1 --repeats 3 \
  --output-dir results/matrix/ml_s_bpr
```

Run the reproducible synthetic benchmark:

```sh
./standalone/star_sampling \
  --dims 48,64,56 --rank 8 --top 1,10,100,1000 \
  --samples 1000,10000,100000 --repeats 5 --seed 2017 \
  --output-dir results/standalone
```

Recreate the three-panel ML-S recall curve from learned factors and render it:

```sh
./standalone/star_sampling \
  --factor-dir results/factors/ml_s_bpr \
  --top 1,10,100,1000 --samples 10000,100000,1000000,10000000 \
  --repeats 10 --seed 2017 \
  --output-dir results/tensor/ml_s_bpr_paper_curve
python3 standalone/plot_basic_recall.py \
  results/tensor/ml_s_bpr_paper_curve/benchmark.csv \
  results/tensor/ml_s_bpr_paper_curve/basic_recall.svg
```

To use learned factor matrices, place row-major matrices in `A.csv`, `B.csv`, and
`C.csv` (one tensor-mode row per line and one latent component per column), then
pass `--factor-dir DIR`. All three matrices must have the same number of columns.

`benchmark.csv` contains one row per top value and random repeat. `truth.csv`
contains the exact top entries, and `metadata.txt` records the input shape, rank,
seed, and exhaustive-search time.

The paper's exact five-data-set plots cannot be regenerated from this repository
alone: it does not contain the preprocessed observations, the trained CP factor
matrices, the train/test split, or the random seeds used for those plots.
