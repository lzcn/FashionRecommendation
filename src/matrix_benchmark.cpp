#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  double operator()(std::size_t i, std::size_t r) const { return values[i * cols + r]; }
};

struct Entry {
  std::uint64_t index = 0;
  double value = 0.0;
};

struct Result {
  std::vector<Entry> top;
  double initialization = 0.0;
  double sampling_scoring = 0.0;
  double filtering = 0.0;
  std::size_t candidates = 0;
  std::size_t actual_samples = 0;
  double total() const { return initialization + sampling_scoring + filtering; }
};

struct Options {
  fs::path factor_dir;
  fs::path output_dir = "results/matrix_benchmark";
  std::size_t users = 1;
  std::vector<std::size_t> samples{10000, 100000};
  std::vector<std::size_t> tops{1, 10, 100, 1000};
  std::vector<std::string> methods{"diamond", "wedge", "core1", "core2"};
  std::size_t repeats = 3;
  double budget_ratio = 0.1;
  std::uint64_t seed = 2017;
  bool self_test = false;
};

static double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static int sign(double x) { return x < 0.0 ? -1 : 1; }

static Matrix load_matrix(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  Matrix matrix;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    std::vector<double> values;
    double value = 0.0;
    while (row >> value) values.push_back(value);
    if (matrix.cols == 0) matrix.cols = values.size();
    if (values.size() != matrix.cols) throw std::runtime_error("ragged matrix");
    matrix.values.insert(matrix.values.end(), values.begin(), values.end());
    ++matrix.rows;
  }
  return matrix;
}

static Matrix user_item_matrix(const Matrix& users, const Matrix& items,
                               std::size_t user) {
  Matrix result{items.rows, items.cols, std::vector<double>(items.values.size())};
  for (std::size_t i = 0; i < items.rows; ++i) {
    for (std::size_t r = 0; r < items.cols; ++r) {
      result.values[i * result.cols + r] = users(user, r) * items(i, r);
    }
  }
  return result;
}

static double dot(const Matrix& left, const Matrix& right, std::size_t i,
                  std::size_t j) {
  double value = 0.0;
  for (std::size_t r = 0; r < left.cols; ++r) value += left(i, r) * right(j, r);
  return value;
}

static std::vector<Entry> exact_top(const Matrix& left, const Matrix& right,
                                    std::size_t top) {
  struct MinValue {
    bool operator()(const Entry& x, const Entry& y) const { return x.value > y.value; }
  };
  std::priority_queue<Entry, std::vector<Entry>, MinValue> heap;
  for (std::size_t i = 0; i < left.rows; ++i) {
    for (std::size_t j = 0; j < right.rows; ++j) {
      Entry entry{static_cast<std::uint64_t>(i) * right.rows + j,
                  dot(left, right, i, j)};
      if (heap.size() < top) heap.push(entry);
      else if (entry.value > heap.top().value) {
        heap.pop();
        heap.push(entry);
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

static std::vector<Entry> filter_candidates(
    const Matrix& left, const Matrix& right,
    const std::unordered_map<std::uint64_t, double>& scores, std::size_t budget,
    std::size_t top) {
  std::vector<std::pair<std::uint64_t, double>> ranked(scores.begin(), scores.end());
  std::sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y) {
    return x.second != y.second ? x.second > y.second : x.first < y.first;
  });
  budget = std::min(budget, ranked.size());
  std::vector<Entry> exact;
  exact.reserve(budget);
  for (std::size_t p = 0; p < budget; ++p) {
    const std::size_t i = ranked[p].first / right.rows;
    const std::size_t j = ranked[p].first % right.rows;
    exact.push_back({ranked[p].first, dot(left, right, i, j)});
  }
  std::sort(exact.begin(), exact.end(), [](const Entry& x, const Entry& y) {
    return x.value != y.value ? x.value > y.value : x.index < y.index;
  });
  if (exact.size() > top) exact.resize(top);
  return exact;
}

static std::vector<double> column_norms(const Matrix& matrix) {
  std::vector<double> norms(matrix.cols, 0.0);
  for (std::size_t i = 0; i < matrix.rows; ++i) {
    for (std::size_t r = 0; r < matrix.cols; ++r) norms[r] += std::abs(matrix(i, r));
  }
  return norms;
}

static Result wedge_or_diamond(const Matrix& left, const Matrix& right,
                               bool diamond, std::size_t samples,
                               std::size_t budget, std::size_t top,
                               std::uint64_t seed) {
  Result result;
  auto start = Clock::now();
  const auto right_norm = column_norms(right);
  std::vector<double> row_norm(left.rows, 0.0);
  for (std::size_t i = 0; i < left.rows; ++i) {
    for (std::size_t r = 0; r < left.cols; ++r) row_norm[i] += std::abs(left(i, r));
  }
  std::vector<double> pair_weight(left.rows * left.cols, 0.0);
  for (std::size_t r = 0; r < left.cols; ++r) {
    for (std::size_t i = 0; i < left.rows; ++i) {
      pair_weight[r * left.rows + i] =
          std::abs(left(i, r)) * right_norm[r] * (diamond ? row_norm[i] : 1.0);
    }
  }
  std::discrete_distribution<std::size_t> pair_distribution(pair_weight.begin(),
                                                             pair_weight.end());
  std::vector<std::discrete_distribution<std::size_t>> right_distribution;
  right_distribution.reserve(right.cols);
  for (std::size_t r = 0; r < right.cols; ++r) {
    std::vector<double> weights(right.rows);
    for (std::size_t j = 0; j < right.rows; ++j) weights[j] = std::abs(right(j, r));
    right_distribution.emplace_back(weights.begin(), weights.end());
  }
  std::vector<std::discrete_distribution<std::size_t>> row_distribution;
  if (diamond) {
    row_distribution.reserve(left.rows);
    for (std::size_t i = 0; i < left.rows; ++i) {
      std::vector<double> weights(left.cols);
      for (std::size_t r = 0; r < left.cols; ++r) weights[r] = std::abs(left(i, r));
      row_distribution.emplace_back(weights.begin(), weights.end());
    }
  }
  result.initialization = seconds(start);

  start = Clock::now();
  std::mt19937_64 rng(seed);
  std::unordered_map<std::uint64_t, double> scores;
  scores.reserve(samples);
  for (std::size_t s = 0; s < samples; ++s) {
    const std::size_t pair = pair_distribution(rng);
    const std::size_t r = pair / left.rows;
    const std::size_t i = pair % left.rows;
    const std::size_t j = right_distribution[r](rng);
    double score = sign(left(i, r)) * sign(right(j, r));
    if (diamond) {
      const std::size_t rp = row_distribution[i](rng);
      score *= sign(left(i, rp)) * right(j, rp);
    }
    scores[static_cast<std::uint64_t>(i) * right.rows + j] += score;
  }
  result.actual_samples = samples;
  result.candidates = scores.size();
  result.sampling_scoring = seconds(start);
  start = Clock::now();
  result.top = filter_candidates(left, right, scores, budget, top);
  result.filtering = seconds(start);
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

static std::vector<double> compound_edge(const Matrix& matrix,
                                         const std::vector<std::size_t>& tuple) {
  std::vector<double> edge(matrix.rows, 1.0);
  for (std::size_t i = 0; i < matrix.rows; ++i) {
    for (std::size_t r : tuple) edge[i] *= matrix(i, r);
  }
  return edge;
}

static Result core_sample(const Matrix& left, const Matrix& right, unsigned power,
                          std::size_t samples, std::size_t budget,
                          std::size_t top, std::uint64_t seed) {
  Result result;
  auto start = Clock::now();
  std::size_t core_count = 1;
  for (unsigned p = 0; p < power; ++p) core_count *= left.cols;
  std::vector<double> weights(core_count, 0.0);
  double total_weight = 0.0;
  for (std::size_t code = 0; code < core_count; ++code) {
    const auto tuple = core_tuple(code, left.cols, power);
    const auto a = compound_edge(left, tuple);
    const auto b = compound_edge(right, tuple);
    const double an = std::accumulate(a.begin(), a.end(), 0.0,
        [](double s, double x) { return s + std::abs(x); });
    const double bn = std::accumulate(b.begin(), b.end(), 0.0,
        [](double s, double x) { return s + std::abs(x); });
    weights[code] = an * bn;
    total_weight += weights[code];
  }
  result.initialization = seconds(start);
  if (!(total_weight > 0.0)) throw std::runtime_error("all core weights are zero");

  start = Clock::now();
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::unordered_map<std::uint64_t, double> scores;
  scores.reserve(samples);
  for (std::size_t code = 0; code < core_count; ++code) {
    const double expected = samples * weights[code] / total_weight;
    const std::size_t frequency = static_cast<std::size_t>(std::floor(expected)) +
        (uniform(rng) < expected - std::floor(expected) ? 1u : 0u);
    if (frequency == 0) continue;
    result.actual_samples += frequency;
    const auto tuple = core_tuple(code, left.cols, power);
    auto a = compound_edge(left, tuple);
    auto b = compound_edge(right, tuple);
    for (double& x : a) x = std::abs(x);
    for (double& x : b) x = std::abs(x);
    std::discrete_distribution<std::size_t> da(a.begin(), a.end());
    std::discrete_distribution<std::size_t> db(b.begin(), b.end());
    for (std::size_t s = 0; s < frequency; ++s) {
      const std::size_t i = da(rng);
      const std::size_t j = db(rng);
      int score = 1;
      for (std::size_t r : tuple) score *= sign(left(i, r)) * sign(right(j, r));
      scores[static_cast<std::uint64_t>(i) * right.rows + j] += score;
    }
  }
  result.candidates = scores.size();
  result.sampling_scoring = seconds(start);
  start = Clock::now();
  result.top = filter_candidates(left, right, scores, budget, top);
  result.filtering = seconds(start);
  return result;
}

static double recall_at(const std::vector<Entry>& truth,
                        const std::vector<Entry>& estimate, std::size_t top) {
  std::unordered_set<std::uint64_t> expected;
  for (std::size_t i = 0; i < std::min(top, truth.size()); ++i) expected.insert(truth[i].index);
  std::size_t hits = 0;
  for (std::size_t i = 0; i < std::min(top, estimate.size()); ++i) hits += expected.count(estimate[i].index);
  return static_cast<double>(hits) / top;
}

static std::vector<std::size_t> parse_sizes(const std::string& text) {
  std::vector<std::size_t> result;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) result.push_back(std::stoull(token));
  return result;
}

static std::vector<std::string> parse_strings(const std::string& text) {
  std::vector<std::string> result;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) result.push_back(token);
  return result;
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--factor-dir") options.factor_dir = next();
    else if (arg == "--output-dir") options.output_dir = next();
    else if (arg == "--users") options.users = std::stoull(next());
    else if (arg == "--samples") options.samples = parse_sizes(next());
    else if (arg == "--top") options.tops = parse_sizes(next());
    else if (arg == "--methods") options.methods = parse_strings(next());
    else if (arg == "--repeats") options.repeats = std::stoull(next());
    else if (arg == "--budget-ratio") options.budget_ratio = std::stod(next());
    else if (arg == "--seed") options.seed = std::stoull(next());
    else if (arg == "--self-test") options.self_test = true;
    else if (arg == "--help") {
      std::cout << "Usage: matrix_benchmark --factor-dir DIR [options]\n"
                   "  --users 1 --samples 10000,100000 --top 1,10,100,1000\n"
                   "  --methods diamond,wedge,core1,core2,core3 --repeats 3\n"
                   "  --budget-ratio 0.1 --seed 2017 --output-dir DIR\n";
      std::exit(0);
    } else throw std::runtime_error("unknown option: " + arg);
  }
  if (!options.self_test && options.factor_dir.empty()) throw std::runtime_error("--factor-dir is required");
  if (options.users == 0 || options.repeats == 0 || options.samples.empty() ||
      options.tops.empty() || options.budget_ratio <= 0.0) {
    throw std::runtime_error("invalid benchmark option");
  }
  return options;
}

static int self_test() {
  Matrix left{3, 2, {3, 1, 1, 2, 0.5, 0.5}};
  Matrix right{4, 2, {2, 0.9, 1, 1, 0.5, 2, 0.1, 0.1}};
  const auto truth = exact_top(left, right, 3);
  const auto core = core_sample(left, right, 2, 100000, 10000, 3, 7);
  const auto diamond = wedge_or_diamond(left, right, true, 100000, 10000, 3, 7);
  const double core_recall = recall_at(truth, core.top, 3);
  const double diamond_recall = recall_at(truth, diamond.top, 3);
  if (core_recall < 1.0 || diamond_recall < 1.0) {
    std::cerr << "matrix benchmark self-test failed: core2=" << core_recall
              << " diamond=" << diamond_recall << '\n';
    return 1;
  }
  std::cout << "matrix benchmark self-test passed\n";
  return 0;
}

int main(int argc, char** argv) try {
  const Options options = parse_options(argc, argv);
  if (options.self_test) return self_test();
  const Matrix users = load_matrix(options.factor_dir / "A.csv");
  const Matrix items = load_matrix(options.factor_dir / "B.csv");
  const Matrix tags = load_matrix(options.factor_dir / "C.csv");
  if (users.cols != items.cols || users.cols != tags.cols) throw std::runtime_error("rank mismatch");
  const std::size_t user_count = std::min(options.users, users.rows);
  const std::size_t max_top = *std::max_element(options.tops.begin(), options.tops.end());
  if (max_top > items.rows * tags.rows) throw std::runtime_error("top exceeds matrix size");
  fs::create_directories(options.output_dir);
  std::ofstream csv(options.output_dir / "benchmark.csv");
  csv << "user,method,requested_samples,actual_samples,top,recall,factorization_s,"
         "initialization_s,sampling_scoring_s,filtering_s,total_s,candidates,repeat,seed\n";
  for (std::size_t user = 0; user < user_count; ++user) {
    auto start = Clock::now();
    const Matrix left = user_item_matrix(users, items, user);
    const double factorization_time = seconds(start);
    start = Clock::now();
    const auto truth = exact_top(left, tags, max_top);
    const double exact_time = seconds(start);
    for (std::size_t top : options.tops) {
      csv << user << ",exact,0,0," << top << ",1," << factorization_time
          << ",0,0,0," << exact_time << ',' << items.rows * tags.rows
          << ",0," << options.seed << '\n';
    }
    std::cout << "user=" << user << " exact=" << exact_time << " s\n";
    for (const auto& method : options.methods) {
      for (std::size_t samples : options.samples) {
        const std::size_t budget = std::max(max_top, static_cast<std::size_t>(samples * options.budget_ratio));
        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
          const std::uint64_t seed = options.seed + user * 10000019ULL +
                                     samples * 31ULL + repeat * 1009ULL;
          Result result;
          if (method == "diamond") result = wedge_or_diamond(left, tags, true, samples, budget, max_top, seed);
          else if (method == "wedge") result = wedge_or_diamond(left, tags, false, samples, budget, max_top, seed);
          else if (method == "core1") result = core_sample(left, tags, 1, samples, budget, max_top, seed);
          else if (method == "core2") result = core_sample(left, tags, 2, samples, budget, max_top, seed);
          else if (method == "core3") result = core_sample(left, tags, 3, samples, budget, max_top, seed);
          else throw std::runtime_error("unknown method: " + method);
          for (std::size_t top : options.tops) {
            csv << user << ',' << method << ',' << samples << ','
                << result.actual_samples << ',' << top << ','
                << std::setprecision(17) << recall_at(truth, result.top, top)
                << ',' << factorization_time << ',' << result.initialization
                << ',' << result.sampling_scoring << ',' << result.filtering
                << ',' << result.total() << ',' << result.candidates << ','
                << repeat << ',' << seed << '\n';
          }
        }
        std::cout << "  " << method << " samples=" << samples << " done\n";
      }
    }
  }
  std::cout << "wrote " << options.output_dir / "benchmark.csv" << '\n';
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 2;
}
