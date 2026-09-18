# Star Sampling (pure C++)

C++17 reproduction of *Sampling for Approximate Maximum Search in Factorized
Tensor* (IJCAI 2017). Replaces the original MATLAB/MEX code.

Status: refactored; self-tests pass; performance untuned.

## Build

CMake >= 3.21, C++17 compiler.

```sh
cmake --preset default
cmake --build --preset default -j 4
ctest --preset default
```

Presets: `default`, `native` (`-march=native` + LTO), `debug`.

## Tools

| Tool | Purpose |
|---|---|
| `prepare_data` | raw tags -> iterative 5-core tensor |
| `factorize` | rank-r BPR training of A, B, C; `legacy-batch` reproduces the MEX trainer |
| `star_sampling` | CPU Core^1/2/3 (Algorithm 1), exhaustive top-t, prefiltering |
| `matrix_benchmark` | matrix special case (diamond, wedge, core1-3) |

Shared helpers in `src/common.hpp`. All except `prepare_data` support
`--self-test`.

## Workflow

```sh
build/default/prepare_data --input data/raw/ml_s/user_taggedbooks-tids.dat \
  --output data/processed/ml_s/triples.csv --min-count 5
build/default/factorize --input data/processed/ml_s/triples.csv \
  --output-dir results/factors/ml_s --rank 64 --epochs 20 --seed 2017
build/default/star_sampling --factor-dir results/factors/ml_s \
  --top 1,10,100,1000 --samples 10000,100000 --repeats 10 \
  --output-dir results/tensor/ml_s_curve
```

## Layout

```
src/              C++17 tools; common.hpp shared
scripts/          Python plot helpers
data/, results/   local artifacts (untracked)
paper/            manuscript (untracked)
```

## Citing

```bibtex
@inproceedings{lu2017sampling,
  title     = {Sampling for Approximate Maximum Search in Factorized Tensor},
  author    = {Lu, Zhi and Hu, Yang and Zeng, Bing},
  booktitle = {Proc. IJCAI},
  year      = {2017}
}
```

## License

MIT.
