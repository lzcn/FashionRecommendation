## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: validation_v1

## Reproduction verdict

**Verdict: PARTIALLY_REPRODUCIBLE.** The standalone C++17 implementation now runs
the paper's Algorithm 1 on newly learned rank-64 ML-S factors as well as on a
fixed-seed synthetic CP tensor. Four public raw data sets were recovered; the
iterative 5-core dimensions of Delicious, LastFM, and ML-S exactly match Table 1.
The exact values in Figures 3-9 still cannot be verified because the learned CP
factor matrices are absent. The LaTeX archive contains rendered figures, not the
numeric arrays used to produce them.

### Materials checked

- Manuscript LaTeX and PDF, including the experiment settings and Figures 3-9.
- MATLAB MEX/C++ implementations of Core^1, Core^2, Core^3, exhaustive search,
  prefiltering, multi-user sampling, and diamond sampling.
- New standalone implementation in `standalone/star_sampling.cpp`.
- Official GroupLens raw archives and the standalone iterative 5-core converter.
- Deleted experiment drivers and result artifacts found in Git history.
- Ten fixed-seed repeats for each point in `results/scale/benchmark.csv` and
  `results/prefilter/benchmark.csv`.

### Reproduction environment

- Apple clang 21.0.0, arm64 macOS
- C++17, `-O3 -Wall -Wextra -Wpedantic`
- Synthetic nonnegative tensor: 256 x 320 x 280, CP rank 8, top-100
- Seed 2017; 10 repeats per sampling point

Timing is environment-sensitive and is therefore compared only qualitatively.
Recall is stochastic and is compared by the mean over the ten repeats, matching
the repetition count stated in the manuscript.

## Main result comparison

| Claim in the paper | Standalone C++ evidence | Assessment |
|---|---|---|
| Increasing k improves recall at the same sample count | At s=10,000: Core^1 0.005, Core^2 0.246, Core^3 0.913 | Confirmed on synthetic data |
| Recall increases with sample count | Core^2: 0.246 (10^4), 0.908 (10^5), 1.000 (10^6) | Confirmed |
| Core^3 has heavier initialization | At s=10^5: initialization 0.000743 s vs 0.000063 s (Core^2) and 0.000009 s (Core^1) | Confirmed |
| Higher k produces fewer distinct candidates at large s and reduces filtering cost | At s=10^6 the total Core^3 runtime was 0.2146 s, below Core^1 0.2638 s and Core^2 0.2577 s; the CSV preserves candidate counts and phase timings | Consistent |
| Sampling can beat exhaustive search | Exact search took 0.0675 s; Core^3 at s=10^4 took 0.0038 s with 0.913 recall (about 18x faster) | Confirmed for approximate, high-recall search at this scale |
| All sampling runs are orders of magnitude faster than exhaustive search | Core^1 at s=10^6 took 0.2638 s, slower than exact search on this modest tensor | Not universal; the paper's claim depends on its much larger tensors and chosen s |
| Prefiltering trades recall for filtering time | At s=10^5, t'=10^4 reduced filtering from 0.0118 to 0.0061 s for Core^2, while recall fell from 0.908 to 0.714 | Confirmed |

The small default benchmark in `results/standalone/benchmark.csv` also exhibits
the same recall ordering, but exhaustive search is faster there because the full
tensor contains only 172,032 entries. This is expected and demonstrates why a
speed claim must be tied to tensor size and sample budget.

## ML-S three-way tensor rerun

The main `BasicRecall` protocol was rerun on the full 456 x 1,973 x 1,222 ML-S
tensor, not on a fixed-user matrix: s in {10^4,10^5,10^6,10^7}, t in
{1,10,100,1000}, t'=s, and ten seeded repeats. Exact top-1000 search took 29.05
seconds on the reproduction machine. At s=10^7, mean recall was:

| Method | top-1 | top-10 | top-100 | top-1000 |
|---|---:|---:|---:|---:|
| Core^1 | 0.100 | 0.020 | 0.030 | 0.0291 |
| Core^2 | 0.100 | 0.120 | 0.102 | 0.0808 |
| Core^3 | 0.300 | 0.270 | 0.256 | 0.2052 |

The qualitative ordering Core^3 > Core^2 > Core^1 is reproduced at the largest
sample count, but the curve is substantially below the paper. The new factors'
exact top-1 and top-1000 scores are 37.159 and 27.666, versus 81.564 and 43.385
in Table 1. More importantly, their head ratio is only 1.34 versus 1.88 in the
paper, so the new score tensor is much flatter and power sampling concentrates
less strongly. The plotted CSV, SVG, PNG, and PDF are in
`results/tensor/ml_s_bpr_paper_curve/`.

### Decomposition diagnosis

The first C++ run used one sampled negative and an immediate pairwise BPR update,
whereas the released MEX trainer aggregates all positive-negative tag pairs for
each user-item post and divides the gradient by their count before updating. A
new `--optimizer legacy-batch` mode now reproduces that schedule. On ML-S it
evaluates 32,936,580 pairs per epoch, versus 27,026 pairs in the initial sampled
run. After 100 epochs, pair accuracy reached 0.9907 and top-1/top-1000 scores rose
to 47.855/36.833, but their ratio remained only 1.30, still far below the paper's
1.88. Thus insufficient negative coverage was a real implementation mismatch,
but it does not fully explain the paper curve. The unpublished BPTF/BPR training
driver, stopping rule, and original factors remain necessary for exact recovery.

The RTF source cited by the manuscript reports zero-mean normal initialization
`N(0,0.1)`, regularization 1e-6, and 500-600 iterations. The released MEX sets
mean 0 and standard deviation 0.1 but calls `MATRIX_INIT_RANDU`; the matrix
implementation ignores those parameters in that branch and samples uniformly
on [0,1]. With the likely intended normal initialization, a 20-epoch diagnostic
produced top-1/top-1000 values 9.442/2.898, a ratio of 3.26, and Core^3 top-1000
recall 0.901 with only 10^4 samples. This overshoots the paper in the opposite
direction and confirms that the decomposition checkpoint, not the Core sampler,
controls the discrepancy. The paper ratio 1.88 has not yet been reproduced.

### High-concentration ML-S diagnostic

The normal-initialized 20-epoch factors (top-1/top-1000 ratio 3.26) were run with
the complete BasicRecall protocol: s in {10^4,10^5,10^6,10^7}, four top-t values,
t'=s, and ten repeats. Core^3 top-1000 recall was 0.9196 at 10^4 samples and 1.0
from 10^5 onward; Core^2 reached 0.7605 at 10^5 and 0.9994 at 10^6. This is much
stronger than the paper and is therefore a sensitivity diagnostic, not a
successful reproduction.

At equal sample counts, Core^1 and Core^2 runtimes changed by at most about 15%
and 7%, respectively. Core^3 was 1.42x faster at 10^5, 1.33x at 10^6, and 1.44x
at 10^7. At 10^7 its mean distinct-candidate count fell from 9.92 million to 2.68
million, explaining the lower filtering cost. Exact search was 29.57 seconds
versus 29.05 seconds, a negligible environment-level difference. Recall and
runtime plots plus summaries are in `results/tensor/ml_s_normal20_paper_curve/`.

### Filtering micro-optimization A/B

A single-run, fixed-seed A/B at s=10^7 replaced two full candidate sorts with
an exact top-1000 heap and skipped estimated-score sorting when t'=s retains all
candidates. Requested/actual samples, candidate counts, and recall matched
exactly for every Core power and top-t. Filtering time changed from 2.381 to
0.544 seconds for Core^1 (4.38x), 2.212 to 0.523 seconds for Core^2 (4.23x), and
0.518 to 0.102 seconds for Core^3 (5.07x). Observed total time improved from
5.268 to 3.709 seconds for Core^1 (1.42x) and 5.176 to 3.648 seconds for Core^2
(1.42x). Core^3 total time was 9.138 versus 9.386 seconds because the unchanged
sampling phase was slower in the second one-shot run; its filtering phase still
improved by 5.07x. Results are under `results/optimization/filter_baseline/` and
`results/optimization/filter_optimized/`.

## Exact-paper comparison status

| Paper output | Status | Missing input |
|---|---|---|
| Table 1 tensor sizes | Exact for Delicious, LastFM, and ML-S; ML-L remains a near match | Exact ML-L snapshot/tag normalization |
| Table 1 top-1 and top-1000 values | Recomputed for new factors; 37.159 and 27.666 do not match 81.564 and 43.385 | Original learned CP factors |
| Figure 3 recall curves | Protocol rerun on new ML-S factors; qualitative ordering matches, numeric curve does not | Original learned CP factors and seeds |
| Figures 4-6 timings | Cannot compare numerically | Factors, original hardware/compiler/runtime, raw timing arrays |
| Figure 7 multi-user result | Cannot rerun | LastFM factors and user order |
| Figure 8 diamond comparison | Cannot rerun | ML-L factors and raw benchmark configuration |
| Figure 9 score estimate | Cannot rerun on ML-S | ML-S factors and raw sampled-score output |

To reach `REPRODUCIBLE`, provide the trained `A`, `B`, and `C` factor matrices for
each data set (or the complete deterministic training configuration), plus the
original seeds. The experiment settings were recovered from Git history, and the
standalone executable accepts the three matrices as `A.csv`, `B.csv`, and `C.csv`
through `--factor-dir`.

## Public data recovery

Official HetRec 2011 Delicious, LastFM, MovieLens v2, MovieLens 10M, and MovieLens
20M archives were downloaded and checksum-tested. Standard iterative 5-core
filtering produced:

| Data | Recovered dimensions | Paper dimensions | Result |
|---|---:|---:|---|
| Delicious | 1,681 x 29,540 x 7,251 | 1,681 x 29,540 x 7,251 | Exact |
| LastFM | 1,348 x 6,927 x 2,132 | 1,348 x 6,927 x 2,132 | Exact |
| ML-S | 456 x 1,973 x 1,222 | 456 x 1,973 x 1,222 | Exact |
| MovieLens 10M | 990 x 3,247 x 2,566 | 993 x 3,298 x 2,555 | Near match |
| MovieLens 20M | 3,432 x 8,971 x 7,979 | 993 x 3,298 x 2,555 | Not ML-L |

This confirms the original raw sources and preprocessing rule for three of the
four in-scope tag data sets. DBLP was excluded at the user's request. Full source
URLs, hashes, commands, and retained-row counts are in `standalone/DATASETS.md`.

## Implementation and manuscript findings

1. Equation (8) in the LaTeX gives the probability of rounding up as
   `floor(mu)-mu`, which is negative. The MEX source implements the intended
   `mu-floor(mu)`, and the standalone implementation follows that behavior.
2. The proof of Theorem 1 says the sample-size condition implies
   `mu <= 3 log(2/sigma)/delta^2`; the displayed assumption actually implies
   `mu >= 3 log(2/sigma)/delta^2`. The theorem statement is consistent with the
   standard Chernoff argument; only this proof line has the reversed inequality.
3. The released code seeds with wall-clock time and the manuscript reports only
   ten-run averages, so the original plotted random sequence is not recoverable.
4. The current checkout omitted the experiment drivers, but they remain in Git
   history. They confirm sample counts of 10^3 through 10^8, top-t in
   {1,10,100,1000}, maximum budget t'=s, and ten repeats for the main curves.
5. No committed version contains the real-data factor matrices. The released
   trainer also omits the complete training driver and fixed seeds, so those
   matrices cannot be reconstructed byte-for-byte from the repository alone.

## Statistical and methodological checks

The paper reports descriptive recall and timing curves, not p-values, confidence
intervals, or formal effect sizes. Consequently, statistical significance cannot
be assessed from the supplied plotted means. Variance/error bars and machine
details would make the timing comparison more defensible.

### Fallacy scan

- Coverage: 11/11 checked.
- Simpson's paradox: not applicable; no subgroup/aggregate inference is reported.
- Ecological fallacy: not detected.
- Berkson's paradox: not applicable to this algorithm benchmark.
- Collider bias: not applicable.
- Base-rate neglect: not applicable.
- Regression to the mean: not applicable.
- Survivorship bias: not detected; failed runs are not described, so exclusion
  handling remains unknown.
- Look-elsewhere effect: no significance testing; data-set and metric selection
  are not presented as inferential tests.
- Garden of forking paths: caution - preprocessing/training and benchmark-driver
  choices are not fully archived, leaving researcher degrees of freedom unclear.
- Correlation is not causation: not applicable to the algorithmic comparison.
- Reverse causality: not applicable.

## Verification limits

This report verifies the standalone source by compilation, self-test, exhaustive
ground truth on tractable tensors, and repeated execution. It does not claim that
synthetic results substitute for the five real-data experiments. The proper
status remains `ANALYZED`, not `VERIFIED`, until the original factor matrices or a
complete deterministic training pipeline are available.
