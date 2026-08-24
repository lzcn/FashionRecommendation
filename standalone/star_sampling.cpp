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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Matrix {
  std::size_t rows = 0, cols = 0;
  std::vector<double> values;
  double operator()(std::size_t i, std::size_t r) const {
    return values[i * cols + r];
  }
};

struct Entry {
  std::uint64_t index = 0;
  double value = 0.0;
};

struct Timings {
  double initialization = 0.0;
  double sampling = 0.0;
  double scoring = 0.0;
  double filtering = 0.0;
  double total() const {
    return initialization + sampling + scoring + filtering;
  }
};

struct SampleResult {
  std::vector<Entry> top;
  Timings time;
  std::size_t actual_samples = 0;
  std::size_t candidates = 0;
};

struct Options {
  std::array<std::size_t, 3> dims{48, 64, 56};
  std::size_t rank = 8;
  std::vector<std::size_t> tops{1, 10, 100, 1000};
  std::size_t repeats = 5;
  std::size_t prefilter = 0;
  std::uint64_t seed = 2017;
  std::vector<std::size_t> samples{1000, 10000, 100000};
  std::string factor_dir;
  std::string output_dir = "results/standalone";
  bool self_test = false;
};

static double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static std::uint64_t encode(std::size_t i, std::size_t j, std::size_t k,
                            const std::array<std::size_t, 3>& dims) {
  return (static_cast<std::uint64_t>(i) * dims[1] + j) * dims[2] + k;
}

static std::array<std::size_t, 3> decode(
    std::uint64_t x, const std::array<std::size_t, 3>& dims) {
  const std::size_t k = x % dims[2];
  x /= dims[2];
  const std::size_t j = x % dims[1];
  return {static_cast<std::size_t>(x / dims[1]), j, k};
}

static double cp_value(const std::array<Matrix, 3>& a, std::size_t i,
                       std::size_t j, std::size_t k) {
  double value = 0.0;
  for (std::size_t r = 0; r < a[0].cols; ++r) {
    value += a[0](i, r) * a[1](j, r) * a[2](k, r);
  }
  return value;
}

static Matrix load_matrix(const fs::path& path) {
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

static std::array<Matrix, 3> generate_factors(
    const std::array<std::size_t, 3>& dims, std::size_t rank,
    std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::lognormal_distribution<double> distribution(-0.25, 0.70);
  std::array<Matrix, 3> factors;
  for (std::size_t n = 0; n < 3; ++n) {
    factors[n].rows = dims[n];
    factors[n].cols = rank;
    factors[n].values.resize(dims[n] * rank);
    for (double& x : factors[n].values) x = distribution(rng);
    // A few consistently strong rows make the maximum-search problem nontrivial
    // while retaining a known, reproducible heavy tail.
    for (std::size_t i = 0; i < std::min<std::size_t>(3, dims[n]); ++i) {
      const double boost = 1.8 - 0.25 * static_cast<double>(i);
      for (std::size_t r = 0; r < rank; ++r) {
        factors[n].values[i * rank + r] *= boost;
      }
    }
  }
  return factors;
}

static std::vector<Entry> exact_top(const std::array<Matrix, 3>& factors,
                                    std::size_t top) {
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
  std::sort(result.begin(), result.end(), [](const Entry& x, const Entry& y) {
    return x.value != y.value ? x.value > y.value : x.index < y.index;
  });
  return result;
}

static std::vector<std::size_t> core_tuple(std::size_t code, std::size_t rank,
                                           unsigned power) {
  std::vector<std::size_t> tuple(power);
  for (unsigned p = 0; p < power; ++p) {
    tuple[power - 1 - p] = code % rank;
    code /= rank;
  }
  return tuple;
}

static int sign(double x) { return x < 0.0 ? -1 : 1; }

static std::vector<double> compound_edge(const Matrix& factor,
                                         const std::vector<std::size_t>& tuple) {
  std::vector<double> edge(factor.rows, 1.0);
  for (std::size_t i = 0; i < factor.rows; ++i) {
    for (std::size_t r : tuple) edge[i] *= factor(i, r);
  }
  return edge;
}

static SampleResult core_sample(const std::array<Matrix, 3>& factors,
                                unsigned power, std::size_t requested_samples,
                                std::size_t prefilter, std::size_t top,
                                std::uint64_t seed) {
  if (power < 1 || power > 3) throw std::runtime_error("power must be 1, 2, or 3");
  const std::size_t rank = factors[0].cols;
  const std::array<std::size_t, 3> dims{
      factors[0].rows, factors[1].rows, factors[2].rows};
  std::size_t core_count = 1;
  for (unsigned p = 0; p < power; ++p) core_count *= rank;
  SampleResult result;
  auto stage = Clock::now();
  std::vector<double> weights(core_count, 0.0);
  double total_weight = 0.0;
  for (std::size_t code = 0; code < core_count; ++code) {
    const auto tuple = core_tuple(code, rank, power);
    double weight = 1.0;
    for (std::size_t n = 0; n < 3; ++n) {
      const auto edge = compound_edge(factors[n], tuple);
      const double norm = std::accumulate(
          edge.begin(), edge.end(), 0.0,
          [](double sum, double x) { return sum + std::abs(x); });
      weight *= norm;
    }
    weights[code] = weight;
    total_weight += weight;
  }
  if (!(total_weight > 0.0)) throw std::runtime_error("all compound-core weights are zero");
  result.time.initialization = seconds(stage);

  stage = Clock::now();
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<std::size_t> frequency(core_count, 0);
  for (std::size_t code = 0; code < core_count; ++code) {
    const double expected = requested_samples * weights[code] / total_weight;
    const double base = std::floor(expected);
    frequency[code] = static_cast<std::size_t>(base) +
                      (uniform(rng) < expected - base ? 1u : 0u);
    result.actual_samples += frequency[code];
  }
  std::unordered_map<std::uint64_t, long long> scores;
  scores.reserve(result.actual_samples);
  for (std::size_t code = 0; code < core_count; ++code) {
    if (frequency[code] == 0) continue;
    const auto tuple = core_tuple(code, rank, power);
    std::array<std::vector<double>, 3> absolute_edge;
    for (std::size_t n = 0; n < 3; ++n) {
      absolute_edge[n] = compound_edge(factors[n], tuple);
      for (double& x : absolute_edge[n]) x = std::abs(x);
    }
    std::array<std::discrete_distribution<std::size_t>, 3> distributions{
        std::discrete_distribution<std::size_t>(absolute_edge[0].begin(), absolute_edge[0].end()),
        std::discrete_distribution<std::size_t>(absolute_edge[1].begin(), absolute_edge[1].end()),
        std::discrete_distribution<std::size_t>(absolute_edge[2].begin(), absolute_edge[2].end())};
    for (std::size_t s = 0; s < frequency[code]; ++s) {
      const std::array<std::size_t, 3> sampled{
          distributions[0](rng), distributions[1](rng), distributions[2](rng)};
      int score = 1;
      for (std::size_t n = 0; n < 3; ++n) {
        for (std::size_t r : tuple) score *= sign(factors[n](sampled[n], r));
      }
      scores[encode(sampled[0], sampled[1], sampled[2], dims)] += score;
    }
  }
  result.time.sampling = seconds(stage);
  result.candidates = scores.size();
  result.time.scoring = 0.0;  // Scoring is fused with sampling to avoid O(s) storage.

  stage = Clock::now();
  const std::size_t budget = prefilter == 0
      ? scores.size() : std::min(prefilter, scores.size());
  const auto better_entry = [](const Entry& x, const Entry& y) {
    return x.value != y.value ? x.value > y.value : x.index < y.index;
  };
  // With BetterEntry as the priority-queue comparator, the worst retained
  // exact entry is at the top and can be replaced in O(log(top)).
  struct BetterEntry {
    bool operator()(const Entry& x, const Entry& y) const {
      return x.value != y.value ? x.value > y.value : x.index < y.index;
    }
  };
  std::priority_queue<Entry, std::vector<Entry>, BetterEntry> top_heap;
  const auto consider = [&](std::uint64_t index) {
    const auto idx = decode(index, dims);
    const Entry entry{index, cp_value(factors, idx[0], idx[1], idx[2])};
    if (top_heap.size() < top) {
      top_heap.push(entry);
    } else if (better_entry(entry, top_heap.top())) {
      top_heap.pop();
      top_heap.push(entry);
    }
  };
  if (budget < scores.size()) {
    std::vector<std::pair<std::uint64_t, long long>> ranked(scores.begin(), scores.end());
    const auto better_score = [](const auto& x, const auto& y) {
      return x.second != y.second ? x.second > y.second : x.first < y.first;
    };
    std::nth_element(ranked.begin(), ranked.begin() + budget, ranked.end(),
                     better_score);
    for (std::size_t p = 0; p < budget; ++p) consider(ranked[p].first);
  } else {
    for (const auto& item : scores) consider(item.first);
  }
  result.top.reserve(top_heap.size());
  while (!top_heap.empty()) {
    result.top.push_back(top_heap.top());
    top_heap.pop();
  }
  std::sort(result.top.begin(), result.top.end(), better_entry);
  result.time.filtering = seconds(stage);
  return result;
}

static double recall(const std::vector<Entry>& truth,
                     const std::vector<Entry>& estimate, std::size_t top) {
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

static std::vector<std::size_t> parse_sizes(const std::string& text) {
  std::vector<std::size_t> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) values.push_back(std::stoull(token));
  return values;
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--dims") {
      const auto d = parse_sizes(next());
      if (d.size() != 3) throw std::runtime_error("--dims requires L1,L2,L3");
      std::copy(d.begin(), d.end(), options.dims.begin());
    } else if (arg == "--rank") options.rank = std::stoull(next());
    else if (arg == "--top") options.tops = parse_sizes(next());
    else if (arg == "--repeats") options.repeats = std::stoull(next());
    else if (arg == "--samples") options.samples = parse_sizes(next());
    else if (arg == "--prefilter") options.prefilter = std::stoull(next());
    else if (arg == "--seed") options.seed = std::stoull(next());
    else if (arg == "--factor-dir") options.factor_dir = next();
    else if (arg == "--output-dir") options.output_dir = next();
    else if (arg == "--self-test") options.self_test = true;
    else if (arg == "--help") {
      std::cout << "Usage: star_sampling [--dims L1,L2,L3] [--rank R] [--top t1,t2,...]\n"
                   "  [--samples s1,s2,...] [--repeats N] [--prefilter N]\n"
                   "  [--seed N] [--factor-dir DIR] [--output-dir DIR] [--self-test]\n"
                   "Factor files are DIR/A.csv, DIR/B.csv, and DIR/C.csv.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (options.rank == 0 || options.tops.empty() ||
      *std::min_element(options.tops.begin(), options.tops.end()) == 0 ||
      options.repeats == 0 || options.samples.empty()) {
    throw std::runtime_error("rank, top, repeats, and samples must be positive");
  }
  return options;
}

static bool self_test() {
  const auto factors = generate_factors({12, 13, 11}, 4, 7);
  const auto truth = exact_top(factors, 10);
  if (truth.size() != 10 || !std::is_sorted(truth.begin(), truth.end(),
      [](const Entry& x, const Entry& y) { return x.value > y.value; })) {
    std::cerr << "self-test failed: exact search ordering\n";
    return false;
  }
  for (unsigned power = 1; power <= 3; ++power) {
    const auto sampled = core_sample(factors, power, 200000, 0, 10, 100 + power);
    if (recall(truth, sampled.top, 10) < 0.9) {
      std::cerr << "self-test failed: Core^" << power << " high-budget recall\n";
      return false;
    }
  }
  std::cout << "self-test passed\n";
  return true;
}

int main(int argc, char** argv) try {
  const Options options = parse_options(argc, argv);
  if (options.self_test) return self_test() ? 0 : 1;

  std::array<Matrix, 3> factors;
  if (options.factor_dir.empty()) {
    factors = generate_factors(options.dims, options.rank, options.seed);
  } else {
    factors = {load_matrix(fs::path(options.factor_dir) / "A.csv"),
               load_matrix(fs::path(options.factor_dir) / "B.csv"),
               load_matrix(fs::path(options.factor_dir) / "C.csv")};
    if (factors[0].cols != factors[1].cols || factors[0].cols != factors[2].cols) {
      throw std::runtime_error("factor matrices must have the same rank");
    }
  }
  const std::array<std::size_t, 3> dims{
      factors[0].rows, factors[1].rows, factors[2].rows};
  const std::size_t total_entries = dims[0] * dims[1] * dims[2];
  const std::size_t max_top = *std::max_element(options.tops.begin(), options.tops.end());
  if (max_top > total_entries) throw std::runtime_error("--top exceeds tensor size");

  fs::create_directories(options.output_dir);
  const fs::path csv_path = fs::path(options.output_dir) / "benchmark.csv";
  std::ofstream csv(csv_path);
  if (!csv) throw std::runtime_error("cannot write " + csv_path.string());
  csv << "power,requested_samples,actual_samples,top,recall,initialization_s,sampling_s,"
         "scoring_s,filtering_s,total_s,candidates,repeat,seed\n";

  std::cout << "Tensor: " << dims[0] << 'x' << dims[1] << 'x' << dims[2]
            << ", rank=" << factors[0].cols << ", max_top=" << max_top << '\n';
  auto exact_start = Clock::now();
  const auto truth = exact_top(factors, max_top);
  const double exact_seconds = seconds(exact_start);
  std::ofstream truth_csv(fs::path(options.output_dir) / "truth.csv");
  truth_csv << "rank,index,value\n";
  for (std::size_t i = 0; i < truth.size(); ++i) {
    truth_csv << i + 1 << ',' << truth[i].index << ','
              << std::setprecision(17) << truth[i].value << '\n';
  }
  std::cout << "Exact search: " << std::fixed << std::setprecision(6)
            << exact_seconds << " s\n";

  for (unsigned power = 1; power <= 3; ++power) {
    for (std::size_t requested : options.samples) {
      double recall_sum = 0.0, time_sum = 0.0;
      for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
        const std::uint64_t seed = options.seed + power * 1000003ULL +
                                   requested * 31ULL + repeat;
        const std::size_t budget = options.prefilter == 0 ? requested : options.prefilter;
        const auto result = core_sample(factors, power, requested, budget,
                                        max_top, seed);
        const double summary_recall = recall(truth, result.top, max_top);
        recall_sum += summary_recall;
        time_sum += result.time.total();
        for (std::size_t top : options.tops) {
          csv << power << ',' << requested << ',' << result.actual_samples << ','
              << top << ',' << std::setprecision(17)
              << recall(truth, result.top, top) << ','
              << result.time.initialization << ',' << result.time.sampling << ','
              << result.time.scoring << ',' << result.time.filtering << ','
              << result.time.total() << ',' << result.candidates << ','
              << repeat << ',' << seed << '\n';
        }
      }
      std::cout << "Core^" << power << " samples=" << requested
                << " recall=" << std::setprecision(4)
                << recall_sum / options.repeats << " time="
                << std::setprecision(6) << time_sum / options.repeats << " s\n";
    }
  }
  std::ofstream meta(fs::path(options.output_dir) / "metadata.txt");
  meta << "source=" << (options.factor_dir.empty() ? "synthetic" : options.factor_dir) << '\n'
       << "dims=" << dims[0] << ',' << dims[1] << ',' << dims[2] << '\n'
       << "rank=" << factors[0].cols << '\n'
       << "tops=";
  for (std::size_t i = 0; i < options.tops.size(); ++i) {
    if (i) meta << ',';
    meta << options.tops[i];
  }
  meta << '\n'
       << "repeats=" << options.repeats << '\n'
       << "seed=" << options.seed << '\n'
       << "exact_seconds=" << std::setprecision(17) << exact_seconds << '\n';
  std::cout << "Wrote " << csv_path << '\n';
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 2;
}
