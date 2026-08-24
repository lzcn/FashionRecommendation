# Star Sampling
Source code for "Sampling for approximate maximum search in factorized tensor" IJCAI 2017

The implementations are located in `mex` folder and run `build_matlab.m` complie the source code.

## Standalone C++17 reproduction

A MATLAB-independent implementation and reproducible benchmark are available in
[`standalone/`](standalone/). Build and run it with:

```sh
make -C standalone test
./standalone/star_sampling --repeats 10 --output-dir results/standalone
```

See [`standalone/README.md`](standalone/README.md) for factor-matrix input format
and all command-line options. The generated `benchmark.csv` contains recall and
phase timings for Core^1, Core^2, and Core^3.
