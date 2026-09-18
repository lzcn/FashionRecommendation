#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Triple {
  std::uint32_t user = 0;
  std::uint32_t item = 0;
  std::uint32_t tag = 0;
};

struct Post {
  std::uint32_t user = 0;
  std::uint32_t item = 0;
  std::vector<std::uint32_t> positives;
  std::unordered_set<std::uint32_t> positive_set;

  bool is_positive(std::uint32_t tag) const { return positive_set.count(tag) != 0; }
};

struct Options {
  fs::path input;
  fs::path output_dir = "results/factors";
  std::size_t rank = 64;
  std::size_t epochs = 20;
  std::size_t negatives = 1;
  double learning_rate = 0.1;
  double regularization = 1e-6;
  double init_scale = 1.0;
  std::string init_distribution = "mex-randu";
  std::uint64_t seed = 2017;
  std::string loss = "legacy-auc";         // bpr|legacy-auc
  std::string negative_mode = "sampled";   // sampled|all
  std::string optimizer = "legacy-batch";  // pairwise|legacy-batch
  bool self_test = false;
};

static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

static std::uint64_t post_key(std::uint32_t u, std::uint32_t i) {
  return (static_cast<std::uint64_t>(u) << 32) | i;
}

static std::vector<Triple> load_triples(const fs::path& path,
                                        std::array<std::size_t, 3>& dims) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::string line;
  std::getline(in, line);  // header
  std::vector<Triple> triples;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto c1 = line.find(',');
    const auto c2 = c1 == std::string::npos ? c1 : line.find(',', c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos) {
      throw std::runtime_error("invalid dense triple row");
    }
    Triple x{static_cast<std::uint32_t>(std::stoul(line.substr(0, c1))),
             static_cast<std::uint32_t>(std::stoul(line.substr(c1 + 1, c2 - c1 - 1))),
             static_cast<std::uint32_t>(std::stoul(line.substr(c2 + 1)))};
    triples.push_back(x);
    dims[0] = std::max(dims[0], static_cast<std::size_t>(x.user) + 1);
    dims[1] = std::max(dims[1], static_cast<std::size_t>(x.item) + 1);
    dims[2] = std::max(dims[2], static_cast<std::size_t>(x.tag) + 1);
  }
  // The MEX trainer groups posts through std::map keyed by (user,item), so its
  // update order is ascending (user,item); sorting here reproduces that order.
  std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
    if (a.user != b.user) return a.user < b.user;
    if (a.item != b.item) return a.item < b.item;
    return a.tag < b.tag;
  });
  triples.erase(std::unique(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
                  return a.user == b.user && a.item == b.item && a.tag == b.tag;
                }),
                triples.end());
  if (triples.empty()) throw std::runtime_error("no triples found");
  return triples;
}

static std::vector<Post> make_posts(const std::vector<Triple>& triples) {
  std::vector<Post> posts;
  std::unordered_map<std::uint64_t, std::size_t> index;
  index.reserve(triples.size());
  for (const auto& x : triples) {
    const auto key = post_key(x.user, x.item);
    auto [it, inserted] = index.emplace(key, posts.size());
    if (inserted) posts.push_back({x.user, x.item, {}, {}});
    auto& post = posts[it->second];
    post.positives.push_back(x.tag);
    post.positive_set.insert(x.tag);
  }
  return posts;
}

static Matrix random_matrix(std::size_t rows, std::size_t cols, double scale,
                            const std::string& distribution,
                            std::mt19937_64& rng) {
  Matrix matrix{rows, cols, std::vector<double>(rows * cols)};
  if (distribution == "normal") {
    std::normal_distribution<double> normal(0.0, scale);
    for (double& x : matrix.values) x = normal(rng);
  } else if (distribution == "uniform") {
    std::uniform_real_distribution<double> uniform(0.0, scale);
    for (double& x : matrix.values) x = uniform(rng);
  } else if (distribution == "mex-randu") {
    // The released MEX trainer initializes through MATRIX_INIT_RANDU, i.e.
    // rand()/(RAND_MAX+1) on [0,1), and never calls srand (implicit seed 1).
    std::srand(1);
    for (double& x : matrix.values) x = std::rand() / (RAND_MAX + 1.0);
  } else {
    throw std::runtime_error("unknown init distribution: " + distribution);
  }
  return matrix;
}

static double sigmoid(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double e = std::exp(x);
  return e / (1.0 + e);
}

struct EpochMetrics {
  double objective = 0.0;
  double mean_probability = 0.0;
  double pair_accuracy = 0.0;
  std::uint64_t pairs = 0;
  double seconds = 0.0;
};

static void update_pair(Matrix& users, Matrix& items, Matrix& tags,
                        std::uint32_t u, std::uint32_t i, std::uint32_t pos,
                        std::uint32_t neg, const Options& options,
                        EpochMetrics& metrics) {
  double difference = 0.0;
  for (std::size_t r = 0; r < users.cols; ++r) {
    difference += users(u, r) * items(i, r) * (tags(pos, r) - tags(neg, r));
  }
  const double probability = sigmoid(difference);
  const double coefficient = options.loss == "bpr" ? 1.0 - probability
                                                   : probability * (1.0 - probability);
  for (std::size_t r = 0; r < users.cols; ++r) {
    const double old_u = users(u, r);
    const double old_i = items(i, r);
    const double old_pos = tags(pos, r);
    const double old_neg = tags(neg, r);
    const double pair_product = old_u * old_i;
    users(u, r) += options.learning_rate *
                   (coefficient * old_i * (old_pos - old_neg) -
                    options.regularization * old_u);
    items(i, r) += options.learning_rate *
                   (coefficient * old_u * (old_pos - old_neg) -
                    options.regularization * old_i);
    tags(pos, r) += options.learning_rate *
                    (coefficient * pair_product - options.regularization * old_pos);
    tags(neg, r) += options.learning_rate *
                    (-coefficient * pair_product - options.regularization * old_neg);
  }
  metrics.objective += options.loss == "bpr"
                           ? std::log(std::max(probability, 1e-15))
                           : probability;
  metrics.mean_probability += probability;
  metrics.pair_accuracy += difference > 0.0 ? 1.0 : 0.0;
  ++metrics.pairs;
}

// Reproduce the released MEX trainer (mex/train/factorization.cpp): aggregate
// every positive-negative tag pair of one (user,item) post, normalize by
// z=|T+||T-|, then apply one joint update. The item factor update reads the
// user factor value that was updated one statement earlier, matching the
// sequential dependency in the original source.
static void update_post_legacy_batch(Matrix& users, Matrix& items, Matrix& tags,
                                     const Post& post, const Options& options,
                                     EpochMetrics& metrics) {
  std::vector<std::uint32_t> negatives;
  negatives.reserve(tags.rows - post.positives.size());
  for (std::uint32_t tag = 0; tag < tags.rows; ++tag) {
    if (!post.positive_set.count(tag)) negatives.push_back(tag);
  }
  if (post.positives.empty() || negatives.empty()) return;
  const double z = static_cast<double>(post.positives.size()) * negatives.size();
  std::vector<double> inter_v(users.cols, 0.0);
  std::vector<double> inter_q(users.cols, 0.0);
  std::vector<double> positive_sum(post.positives.size(), 0.0);
  std::vector<double> negative_sum(negatives.size(), 0.0);
  for (std::size_t p = 0; p < post.positives.size(); ++p) {
    const std::uint32_t pos = post.positives[p];
    for (std::size_t n = 0; n < negatives.size(); ++n) {
      const std::uint32_t neg = negatives[n];
      double difference = 0.0;
      for (std::size_t r = 0; r < users.cols; ++r) {
        difference += users(post.user, r) * items(post.item, r) *
                      (tags(pos, r) - tags(neg, r));
      }
      const double probability = sigmoid(difference);
      const double coefficient = probability * (1.0 - probability);
      positive_sum[p] += coefficient;
      negative_sum[n] += coefficient;
      for (std::size_t r = 0; r < users.cols; ++r) {
        inter_v[r] += coefficient * (tags(pos, r) - tags(neg, r));
      }
      metrics.objective += probability;
      metrics.mean_probability += probability;
      metrics.pair_accuracy += difference > 0.0 ? 1.0 : 0.0;
      ++metrics.pairs;
    }
  }
  for (std::size_t r = 0; r < users.cols; ++r) {
    inter_q[r] = users(post.user, r) * items(post.item, r);
  }
  for (std::size_t r = 0; r < users.cols; ++r) {
    // MEX literal order: the item update consumes the freshly updated user
    // value from the statement above.
    users(post.user, r) += options.learning_rate *
        (items(post.item, r) * inter_v[r] / z -
         options.regularization * users(post.user, r));
    items(post.item, r) += options.learning_rate *
        (users(post.user, r) * inter_v[r] / z -
         options.regularization * items(post.item, r));
  }
  for (std::size_t p = 0; p < post.positives.size(); ++p) {
    const auto pos = post.positives[p];
    for (std::size_t r = 0; r < users.cols; ++r) {
      tags(pos, r) += options.learning_rate *
          (inter_q[r] * positive_sum[p] / z -
           options.regularization * tags(pos, r));
    }
  }
  for (std::size_t n = 0; n < negatives.size(); ++n) {
    const auto neg = negatives[n];
    for (std::size_t r = 0; r < users.cols; ++r) {
      tags(neg, r) += options.learning_rate *
          (-inter_q[r] * negative_sum[n] / z -
           options.regularization * tags(neg, r));
    }
  }
}

static std::uint32_t sample_negative(const Post& post, std::size_t tag_count,
                                     std::mt19937_64& rng) {
  if (post.positive_set.size() >= tag_count) {
    throw std::runtime_error("post has no negative tag");
  }
  std::uniform_int_distribution<std::uint32_t> distribution(
      0, static_cast<std::uint32_t>(tag_count - 1));
  std::uint32_t tag = 0;
  do {
    tag = distribution(rng);
  } while (post.is_positive(tag));
  return tag;
}

static void save_matrix(const Matrix& matrix, const fs::path& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << std::setprecision(17);
  for (std::size_t i = 0; i < matrix.rows; ++i) {
    for (std::size_t r = 0; r < matrix.cols; ++r) {
      if (r) out << ',';
      out << matrix(i, r);
    }
    out << '\n';
  }
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--input") options.input = next();
    else if (arg == "--output-dir") options.output_dir = next();
    else if (arg == "--rank") options.rank = std::stoull(next());
    else if (arg == "--epochs") options.epochs = std::stoull(next());
    else if (arg == "--negatives") options.negatives = std::stoull(next());
    else if (arg == "--learning-rate") options.learning_rate = std::stod(next());
    else if (arg == "--regularization") options.regularization = std::stod(next());
    else if (arg == "--init-scale") options.init_scale = std::stod(next());
    else if (arg == "--init-distribution") options.init_distribution = next();
    else if (arg == "--seed") options.seed = std::stoull(next());
    else if (arg == "--loss") options.loss = next();
    else if (arg == "--negative-mode") options.negative_mode = next();
    else if (arg == "--optimizer") options.optimizer = next();
    else if (arg == "--self-test") options.self_test = true;
    else if (arg == "--help") {
      std::cout << "Usage: factorize --input triples.csv --output-dir DIR [options]\n"
                   "  --rank 64 --epochs 20 --negatives 1 --learning-rate 0.1\n"
                   "  --regularization 1e-6 --seed 2017\n"
                   "  --init-distribution mex-randu|uniform|normal\n"
                   "      mex-randu reproduces MATRIX_INIT_RANDU of the released\n"
                   "      MEX trainer: rand()/(RAND_MAX+1) on [0,1), implicit seed.\n"
                   "  --loss bpr|legacy-auc --negative-mode sampled|all\n"
                   "  --optimizer legacy-batch|pairwise\n"
                   "      legacy-batch is the paper's schedule: every positive-\n"
                   "      negative tag pair per post aggregated, divided by z,\n"
                   "      posts visited in ascending (user,item) order.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (!options.self_test && options.input.empty()) throw std::runtime_error("--input is required");
  if (options.rank == 0 || options.epochs == 0 || options.negatives == 0 ||
      options.learning_rate <= 0.0 || options.regularization < 0.0 ||
      options.init_scale <= 0.0) {
    throw std::runtime_error("invalid non-positive training option");
  }
  if (options.loss != "bpr" && options.loss != "legacy-auc") {
    throw std::runtime_error("--loss must be bpr or legacy-auc");
  }
  if (options.negative_mode != "sampled" && options.negative_mode != "all") {
    throw std::runtime_error("--negative-mode must be sampled or all");
  }
  if (options.optimizer != "pairwise" && options.optimizer != "legacy-batch") {
    throw std::runtime_error("--optimizer must be pairwise or legacy-batch");
  }
  if (options.init_distribution != "mex-randu" &&
      options.init_distribution != "uniform" &&
      options.init_distribution != "normal") {
    throw std::runtime_error("--init-distribution must be mex-randu, uniform, or normal");
  }
  return options;
}

static int self_test() {
  Options options;
  options.rank = 4;
  options.epochs = 25;
  options.learning_rate = 0.05;
  options.loss = "bpr";
  std::vector<Triple> triples{{0, 0, 0}, {0, 1, 0}, {1, 0, 1}, {1, 1, 1}};
  const auto posts = make_posts(triples);
  std::mt19937_64 rng(7);
  Matrix users = random_matrix(2, options.rank, 0.1, "uniform", rng);
  Matrix items = random_matrix(2, options.rank, 0.1, "uniform", rng);
  Matrix tags = random_matrix(3, options.rank, 0.1, "uniform", rng);
  double first = 0.0, last = 0.0;
  for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
    EpochMetrics metrics;
    for (const auto& post : posts) {
      for (auto pos : post.positives) {
        const auto neg = sample_negative(post, tags.rows, rng);
        update_pair(users, items, tags, post.user, post.item, pos, neg, options, metrics);
      }
    }
    const double mean = metrics.mean_probability / metrics.pairs;
    if (epoch == 0) first = mean;
    last = mean;
  }
  if (!(last > first) || !std::all_of(users.values.begin(), users.values.end(),
                                      [](double x) { return std::isfinite(x); })) {
    std::cerr << "factorize self-test failed\n";
    return 1;
  }
  EpochMetrics legacy_metrics;
  update_post_legacy_batch(users, items, tags, posts.front(), options,
                           legacy_metrics);
  if (legacy_metrics.pairs != 2 || !std::isfinite(legacy_metrics.objective)) {
    std::cerr << "factorize self-test failed: legacy batch update\n";
    return 1;
  }
  std::cout << "factorize self-test passed: probability " << first << " -> " << last << '\n';
  return 0;
}

int main(int argc, char** argv) try {
  const Options options = parse_options(argc, argv);
  if (options.self_test) return self_test();
  std::array<std::size_t, 3> dims{0, 0, 0};
  const auto triples = load_triples(options.input, dims);
  const auto posts = make_posts(triples);
  std::mt19937_64 rng(options.seed);
  Matrix users = random_matrix(dims[0], options.rank, options.init_scale,
                               options.init_distribution, rng);
  Matrix items = random_matrix(dims[1], options.rank, options.init_scale,
                               options.init_distribution, rng);
  Matrix tags = random_matrix(dims[2], options.rank, options.init_scale,
                              options.init_distribution, rng);
  fs::create_directories(options.output_dir);
  std::ofstream log(options.output_dir / "training.csv");
  log << "epoch,pairs,objective,mean_probability,pair_accuracy,seconds\n";
  std::vector<std::size_t> order(posts.size());
  std::iota(order.begin(), order.end(), 0);

  const bool use_legacy_batch = options.optimizer == "legacy-batch";

  std::cout << "triples=" << triples.size() << " posts=" << posts.size()
            << " dims=" << dims[0] << 'x' << dims[1] << 'x' << dims[2]
            << " rank=" << options.rank
            << " optimizer=" << options.optimizer
            << " init=" << options.init_distribution << '\n';
  for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
    const auto start = Clock::now();
    EpochMetrics metrics;
    {
      if (!use_legacy_batch) {
        std::shuffle(order.begin(), order.end(), rng);
      }
      for (std::size_t post_index : order) {
        const auto& post = posts[post_index];
        if (use_legacy_batch) {
          update_post_legacy_batch(users, items, tags, post, options, metrics);
          continue;
        }
        for (std::uint32_t pos : post.positives) {
          if (options.negative_mode == "all") {
            for (std::uint32_t neg = 0; neg < tags.rows; ++neg) {
              if (!post.is_positive(neg)) {
                update_pair(users, items, tags, post.user, post.item, pos, neg,
                            options, metrics);
              }
            }
          } else {
            for (std::size_t n = 0; n < options.negatives; ++n) {
              const auto neg = sample_negative(post, tags.rows, rng);
              update_pair(users, items, tags, post.user, post.item, pos, neg,
                          options, metrics);
            }
          }
        }
      }
    }
    metrics.seconds = elapsed(start);
    const double divisor = static_cast<double>(metrics.pairs);
    log << (epoch + 1) << ',' << metrics.pairs << ',' << std::setprecision(17)
        << metrics.objective / divisor << ',' << metrics.mean_probability / divisor
        << ',' << metrics.pair_accuracy / divisor << ',' << metrics.seconds << '\n';
    std::cout << "epoch=" << (epoch + 1) << " probability="
              << std::setprecision(6) << metrics.mean_probability / divisor
              << " pair_accuracy=" << metrics.pair_accuracy / divisor
              << " seconds=" << metrics.seconds << '\n';
  }
  save_matrix(users, options.output_dir / "A.csv");
  save_matrix(items, options.output_dir / "B.csv");
  save_matrix(tags, options.output_dir / "C.csv");
  std::ofstream metadata(options.output_dir / "metadata.txt");
  metadata << "input=" << options.input << '\n'
           << "dims=" << dims[0] << ',' << dims[1] << ',' << dims[2] << '\n'
           << "rank=" << options.rank << '\n'
           << "epochs=" << options.epochs << '\n'
           << "negatives=" << options.negatives << '\n'
           << "learning_rate=" << options.learning_rate << '\n'
           << "regularization=" << options.regularization << '\n'
           << "init_scale=" << options.init_scale << '\n'
           << "init_distribution=" << options.init_distribution << '\n'
           << "seed=" << options.seed << '\n'
           << "loss=" << options.loss << '\n'
           << "negative_mode=" << options.negative_mode << '\n'
           << "optimizer=" << options.optimizer << '\n';
  std::cout << "wrote factors to " << options.output_dir << '\n';
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 2;
}
