# Star Sampling (pure C++)

Pure C++17 implementation and reproducible benchmark for *Sampling for
Approximate Maximum Search in Factorized Tensor* (IJCAI 2017).

The `cpp` branch replaces the original MATLAB/MEX distribution entirely:
data preparation, CP/BPR factorization, Core^1/Core^2/Core^3 sampling,
prefiltering, exact search, and the matrix special-case baselines are all
standard-library C++ with a single CMake build. The historical MATLAB/MEX
sources remain on the `master` branch.

## Build

Requirements: CMake >= 3.21 and a C++17 compiler.

```sh
cmake --preset default            # configure (build/default)
cmake --build --preset default -j # compile all four tools
ctest --preset default            # run built-in self tests
```

Presets: `default` (portable Release), `native` (`-march=native` + LTO,
use this for timing benchmarks), `debug`.

## Tools

| Tool | Purpose |
|---|---|
| `prepare_data` | raw tag assignments -> iterative 5-core tensor with dense zero-based IDs |
| `factorize` | rank-r BPR pairwise SGD training of user/item/tag factors A, B, C |
| `star_sampling` | Algorithm 1: exhaustive top-t search plus Core^1/2/3 sampling, recall and phase timings, optional prefiltering |
| `matrix_benchmark` | matrix special-case baselines (diamond, wedge, core1-3) used for the paper's Diamond comparison |

Binaries land in `build/<preset>/<tool>`.

## Workflow

```sh
# 1. preprocess a raw dataset (HetRec TSV by default; --format ml10m|ml20m)
build/default/prepare_data --input data/raw/ml_s/user_taggedbooks-tids.dat \
  --output data/processed/ml_s/triples.csv --min-count 5

# 2. train rank-64 factors (BPR, one sampled negative per positive tag)
build/default/factorize --input data/processed/ml_s/triples.csv \
  --output-dir results/factors/ml_s --rank 64 --epochs 20 --seed 2017

# 3. run the BasicRecall protocol on synthetic or learned factors
build/default/star_sampling --factor-dir results/factors/ml_s \
  --top 1,10,100,1000 --samples 10000,100000,1000000,10000000 \
  --repeats 10 --seed 2017 --output-dir results/tensor/ml_s_curve

# 4. render recall/runtime figures from benchmark.csv
python3 scripts/plot_basic_recall.py results/tensor/ml_s_curve/benchmark.csv out.svg
```

`star_sampling --self-test`, `factorize --self-test`, and
`matrix_benchmark --self-test` verify each tool against exhaustive ground
truth on small tensors; they run automatically under `ctest`.
`--help` lists all options of each tool.

## Layout

```
CMakeLists.txt   root build script (all tools)
src/             the four C++17 tools
scripts/         Python plot helpers (optional)
docs/            reproduction scope, datasets provenance, validation report
data/, results/  local experiment inputs/outputs (not tracked)
paper/           original LaTeX manuscript (not tracked)
```

## Reproduction status

See [`docs/reproduction_report.md`](docs/reproduction_report.md). Summary:
the qualitative claims (recall ordering Core^3 > Core^2 > Core^1, recall
growing with sample count, prefilter trade-off) are reproduced; the paper's
exact curves depend on the original learned CP factors, whose training
driver was never released. Factor calibration experiments that search for a
training configuration matching the paper's score concentration live in
`results/factors/` once run locally.
