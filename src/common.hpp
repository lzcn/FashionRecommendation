// Shared utilities for the Star Sampling tool suite.
//
// Every tool used to carry private copies of these declarations; they now
// live here so that factor loading/synthesis, exhaustive ground truth, and
// the top-t metric helpers have exactly one definition.
// Everything sits in an unnamed namespace on purpose: each translation unit
// keeps internal linkage, so including this header never creates ODR
// coupling between tools, and [[maybe_unused]] keeps -Wall quiet for the
// tools that exercise only part of the surface.
#ifndef STAR_SAMPLING_COMMON_HPP
#define STAR_SAMPLING_COMMON_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct [[maybe_unused]] Matrix {
  std::size_t rows = 0, cols = 0;
  std::vector<double> values;
  double operator()(std::size_t i, std::size_t r) const {
    return values[i * cols + r];
  }
  double& operator()(std::size_t i, std::size_t r) {
    return values[i * cols + r];
  }
};

struct [[maybe_unused]] Entry {
  std::uint64_t index = 0;
  double value = 0.0;
};

// Descending by value, ascending by index on ties: gives deterministic,
// reproducible top-t lists even for repeated scores.
struct [[maybe_unused]] BetterEntry {
  bool operator()(const Entry& x, const Entry& y) const {
    return x.value != y.value ? x.value > y.value : x.index < y.index;
  }
};

struct [[maybe_unused]] Timings {
  double initialization = 0.0;
  double sampling = 0.0;
  double scoring = 0.0;
  double filtering = 0.0;
  double total() const {
    return initialization + sampling + scoring + filtering;
  }
};

[[maybe_unused]] static double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[maybe_unused]] static std::uint64_t encode(std::size_t i, std::size_t j,
                                             std::size_t k,
                                             const std::array<std::size_t, 3>& dims) {
  return (static_cast<std::uint64_t>(i) * dims[1] + j) * dims[2] + k;
}

[[maybe_unused]] static std::array<std::size_t, 3> decode(
    std::uint64_t x, const std::array<std::size_t, 3>& dims) {
  const std::size_t k = x % dims[2];
  x /= dims[2];
  const std::size_t j = x % dims[1];
  return {static_cast<std::size_t>(x / dims[1]), j, k};
}

[[maybe_unused]] static double cp_value(const std::array<Matrix, 3>& a,
                                        std::size_t i, std::size_t j,
                                        std::size_t k) {
  double value = 0.0;
  for (std::size_t r = 0; r < a[0].cols; ++r) {
    value += a[0](i, r) * a[1](j, r) * a[2](k, r);
  }
  return value;
}

[[maybe_unused]] static Matrix load_matrix(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open factor matrix: " + path.string());
  Matrix out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    std::vector<double> parsed;
    double value = 0.0;
    while (row >> value) parsed.push_back(value);
    if (parsed.empty()) continue;
    if (out.cols == 0) out.cols = parsed.size();
    if (parsed.size() != out.cols) {
      throw std::runtime_error("ragged factor matrix: " + path.string());
    }
    out.values.insert(out.values.end(), parsed.begin(), parsed.end());
    ++out.rows;
  }
  if (out.rows == 0 || out.cols == 0) {
    throw std::runtime_error("empty factor matrix: " + path.string());
  }
  return out;
}

// Loads the CP triplet A.csv/B.csv/C.csv from a factor directory produced by
// the `factorize` tool.
[[maybe_unused]] static std::array<Matrix, 3> load_factors(
    const std::filesystem::path& dir) {
  return {load_matrix(dir / "A.csv"), load_matrix(dir / "B.csv"),
          load_matrix(dir / "C.csv")};
}

// Canonical synthetic factor generator shared by the benchmark tools. The
// MT19937_64 stream is seeded once and carried across modes; the first three
// rows per mode receive a decreasing boost so the maximum-search problem has
// a known heavy tail.
[[maybe_unused]] static std::array<Matrix, 3> generate_factors(
    const std::array<std::size_t, 3>& dims, std::size_t rank,
    std::uint64_t seed, bool signed_values = false) {
  std::mt19937_64 rng(seed);
  std::lognormal_distribution<double> distribution(-0.25, 0.70);
  std::uniform_int_distribution<int> flip(0, 1);
  std::array<Matrix, 3> factors;
  for (std::size_t n = 0; n < 3; ++n) {
    factors[n].rows = dims[n];
    factors[n].cols = rank;
    factors[n].values.resize(dims[n] * rank);
    for (double& x : factors[n].values) {
      x = distribution(rng);
      if (signed_values && flip(rng)) x = -x;
    }
    for (std::size_t i = 0; i < std::min<std::size_t>(3, dims[n]); ++i) {
      const double boost = 1.8 - 0.25 * static_cast<double>(i);
      for (std::size_t r = 0; r < rank; ++r) {
        factors[n].values[i * rank + r] *= boost;
      }
    }
  }
  return factors;
}

// Exhaustive top-t search used as ground truth on small tensors.
[[maybe_unused]] static std::vector<Entry> exact_top(
    const std::array<Matrix, 3>& factors, std::size_t top) {
  struct MinValue {
    bool operator()(const Entry& x, const Entry& y) const {
      return x.value > y.value;
    }
  };
  std::priority_queue<Entry, std::vector<Entry>, MinValue> heap;
  const std::array<std::size_t, 3> dims{
      factors[0].rows, factors[1].rows, factors[2].rows};
  for (std::size_t i = 0; i < dims[0]; ++i) {
    for (std::size_t j = 0; j < dims[1]; ++j) {
      for (std::size_t k = 0; k < dims[2]; ++k) {
        Entry e{encode(i, j, k, dims), cp_value(factors, i, j, k)};
        if (heap.size() < top) {
          heap.push(e);
        } else if (e.value > heap.top().value) {
          heap.pop();
          heap.push(e);
        }
      }
    }
  }
  std::vector<Entry> result;
  while (!heap.empty()) {
    result.push_back(heap.top());
    heap.pop();
  }
  std::sort(result.begin(), result.end(), BetterEntry());
  return result;
}

[[maybe_unused]] static double recall(const std::vector<Entry>& truth,
                                      const std::vector<Entry>& estimate,
                                      std::size_t top) {
  std::unordered_set<std::uint64_t> expected;
  for (std::size_t i = 0; i < std::min(top, truth.size()); ++i) {
    expected.insert(truth[i].index);
  }
  std::size_t hits = 0;
  for (std::size_t i = 0; i < std::min(top, estimate.size()); ++i) {
    hits += expected.count(estimate[i].index);
  }
  return top == 0 ? 1.0 : static_cast<double>(hits) / top;
}

[[maybe_unused]] static std::vector<std::size_t> parse_sizes(
    const std::string& text) {
  std::vector<std::size_t> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) values.push_back(std::stoull(token));
  return values;
}

// Core^k sampling primitives shared by the star_sampling tool (3-mode
// tensor) and the matrix_benchmark baseline (per-slice matrix case).

[[maybe_unused]] static std::vector<std::size_t> core_tuple(std::size_t code,
                                                            std::size_t rank,
                                                            unsigned power) {
  std::vector<std::size_t> tuple(power);
  for (unsigned p = 0; p < power; ++p) {
    tuple[power - 1 - p] = code % rank;
    code /= rank;
  }
  return tuple;
}

[[maybe_unused]] static int sign(double x) { return x < 0.0 ? -1 : 1; }

[[maybe_unused]] static std::vector<double> compound_edge(
    const Matrix& factor, const std::vector<std::size_t>& tuple) {
  std::vector<double> edge(factor.rows, 1.0);
  for (std::size_t i = 0; i < factor.rows; ++i) {
    for (std::size_t r : tuple) edge[i] *= factor(i, r);
  }
  return edge;
}

[[maybe_unused]] static bool top_lists_match(const std::vector<Entry>& truth,
                                             const std::vector<Entry>& estimate,
                                             double tol) {
  if (truth.size() != estimate.size()) return false;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    if (truth[i].index != estimate[i].index ||
        std::abs(truth[i].value - estimate[i].value) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace

#endif  // STAR_SAMPLING_COMMON_HPP
