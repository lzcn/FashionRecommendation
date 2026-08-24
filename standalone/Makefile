CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
TARGETS := star_sampling prepare_data factorize matrix_benchmark

.PHONY: all test clean

all: $(TARGETS)

star_sampling: star_sampling.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

prepare_data: prepare_data.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

factorize: factorize.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

matrix_benchmark: matrix_benchmark.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test: star_sampling prepare_data factorize matrix_benchmark
	./star_sampling --self-test
	./factorize --self-test
	./matrix_benchmark --self-test

clean:
	rm -f $(TARGETS)
