#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

struct Triple {
  std::string user;
  std::string item;
  std::string tag;
  bool active = true;
};

static std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const auto end = line.find('\t', start);
    fields.push_back(line.substr(start, end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

static Triple parse_ml10m(const std::string& line) {
  const auto first = line.find("::");
  const auto second = first == std::string::npos ? first : line.find("::", first + 2);
  const auto last = line.rfind("::");
  if (first == std::string::npos || second == std::string::npos || last <= second) {
    throw std::runtime_error("invalid MovieLens 10M tag row");
  }
  return {line.substr(0, first), line.substr(first + 2, second - first - 2),
          line.substr(second + 2, last - second - 2), true};
}

static Triple parse_ml20m(const std::string& line) {
  const auto first = line.find(',');
  const auto second = first == std::string::npos ? first : line.find(',', first + 1);
  const auto last = line.rfind(',');
  if (first == std::string::npos || second == std::string::npos || last <= second) {
    throw std::runtime_error("invalid MovieLens 20M tag row");
  }
  std::string tag = line.substr(second + 1, last - second - 1);
  if (tag.size() >= 2 && tag.front() == '"' && tag.back() == '"') {
    tag = tag.substr(1, tag.size() - 2);
    std::size_t pos = 0;
    while ((pos = tag.find("\"\"", pos)) != std::string::npos) {
      tag.replace(pos, 2, "\"");
      ++pos;
    }
  }
  return {line.substr(0, first), line.substr(first + 1, second - first - 1),
          std::move(tag), true};
}

int main(int argc, char** argv) try {
  fs::path input, output;
  std::string format = "tsv";
  std::size_t minimum = 5;
  bool lowercase_tags = false;
  bool legacy_mex_core = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--input") input = next();
    else if (arg == "--output") output = next();
    else if (arg == "--format") format = next();
    else if (arg == "--min-count") minimum = std::stoull(next());
    else if (arg == "--lowercase-tags") lowercase_tags = true;
    else if (arg == "--legacy-mex-core") legacy_mex_core = true;
    else if (arg == "--help") {
      std::cout << "Usage: prepare_data --input FILE --output FILE "
                   "[--format tsv|ml10m|ml20m] [--min-count 5] "
                   "[--lowercase-tags] [--legacy-mex-core]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (input.empty() || output.empty() || minimum == 0) {
    throw std::runtime_error("--input, --output, and a positive --min-count are required");
  }
  if (format != "tsv" && format != "ml10m" && format != "ml20m") {
    throw std::runtime_error("--format must be tsv, ml10m, or ml20m");
  }

  std::ifstream in(input);
  if (!in) throw std::runtime_error("cannot open " + input.string());
  std::vector<Triple> triples;
  std::string line;
  if (format == "tsv" || format == "ml20m") std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (format == "ml10m") {
      triples.push_back(parse_ml10m(line));
    } else if (format == "ml20m") {
      triples.push_back(parse_ml20m(line));
    } else {
      const auto fields = split_tabs(line);
      if (fields.size() < 3) throw std::runtime_error("invalid TSV row");
      triples.push_back({fields[0], fields[1], fields[2], true});
    }
    if (lowercase_tags) {
      auto& tag = triples.back().tag;
      std::transform(tag.begin(), tag.end(), tag.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
    }
  }

  std::size_t iteration = 0;
  if (legacy_mex_core) {
    std::unordered_map<std::string, std::size_t> users, items, tags;
    std::set<std::tuple<std::string, std::string, std::string>> unique;
    for (const auto& x : triples) {
      ++users[x.user];
      ++items[x.item];
      ++tags[x.tag];
      unique.emplace(x.user, x.item, x.tag);
    }
    triples.clear();
    triples.reserve(unique.size());
    for (const auto& x : unique) {
      triples.push_back({std::get<0>(x), std::get<1>(x), std::get<2>(x), true});
    }
    while (true) {
      std::size_t removed = 0;
      for (auto& x : triples) {
        if (!x.active) continue;
        if (users[x.user] < minimum || items[x.item] < minimum ||
            tags[x.tag] < minimum) {
          x.active = false;
          --users[x.user];
          --items[x.item];
          --tags[x.tag];
          ++removed;
        }
      }
      ++iteration;
      if (removed == 0) break;
    }
  } else {
    while (true) {
      std::unordered_map<std::string, std::size_t> users, items, tags;
      for (const auto& x : triples) {
        if (!x.active) continue;
        ++users[x.user];
        ++items[x.item];
        ++tags[x.tag];
      }
      std::size_t removed = 0;
      for (auto& x : triples) {
        if (!x.active) continue;
        if (users[x.user] < minimum || items[x.item] < minimum ||
            tags[x.tag] < minimum) {
          x.active = false;
          ++removed;
        }
      }
      ++iteration;
      if (removed == 0) break;
    }
  }

  std::unordered_map<std::string, std::size_t> users, items, tags;
  auto dense_id = [](auto& map, const std::string& key) {
    const auto found = map.find(key);
    if (found != map.end()) return found->second;
    const std::size_t id = map.size();
    map.emplace(key, id);
    return id;
  };
  if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
  std::ofstream out(output);
  if (!out) throw std::runtime_error("cannot write " + output.string());
  out << "user,item,tag\n";
  std::size_t retained = 0;
  for (const auto& x : triples) {
    if (!x.active) continue;
    out << dense_id(users, x.user) << ',' << dense_id(items, x.item) << ','
        << dense_id(tags, x.tag) << '\n';
    ++retained;
  }
  std::cout << "rows=" << retained << " users=" << users.size()
            << " items=" << items.size() << " tags=" << tags.size()
            << " iterations=" << iteration << '\n';
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 2;
}
