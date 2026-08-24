# Dataset recovery notes

The public archives are stored under `data/raw/` and are intentionally ignored by
Git. They were downloaded from the official GroupLens file server on 2026-07-22.

| Archive | Official URL | MD5 |
|---|---|---|
| HetRec Delicious 2k | https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip | `a4b4a3b1629f1291fc5f414cb649999d` |
| HetRec LastFM 2k | https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip | `296d61afe4e8632b173fc2dd3be20ce2` |
| HetRec MovieLens 2k v2 | https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip | `3b714d797f1cd512a6676a6e59ab9efc` |
| MovieLens 10M | https://files.grouplens.org/datasets/movielens/ml-10m.zip | `ce571fd55effeba0271552578f2648bd` |
| MovieLens 20M | https://files.grouplens.org/datasets/movielens/ml-20m.zip | `cd245b17a1ae2cc31bb14903e1204af3` |

All five archives pass `unzip -t`. The MovieLens 10M and 20M hashes match the
official `.md5` files.

## Iterative 5-core results

The standalone preprocessor repeatedly removes tag assignments whose user, item,
or tag has fewer than five remaining occurrences. Dense triples are written under
`data/processed/`.

| Input | Retained rows | Users | Items | Tags | Paper dimensions |
|---|---:|---:|---:|---:|---:|
| Delicious | 283,030 | 1,681 | 29,540 | 7,251 | exact match |
| LastFM | 162,047 | 1,348 | 6,927 | 2,132 | exact match |
| HetRec MovieLens (ML-S) | 27,026 | 456 | 1,973 | 1,222 | exact match |
| MovieLens 10M | 61,688 | 990 | 3,247 | 2,566 | paper ML-L: 993 x 3,298 x 2,555 |
| MovieLens 20M | 390,952 | 3,432 | 8,971 | 7,979 | not ML-L |

The repository's deleted experiment scripts referenced both MovieLens 10M and
20M. The dimensions show that the paper's ML-L is closest to 10M, but its exact
tag snapshot or normalization procedure is not archived. Lowercasing MovieLens
10M tags gives 1,030 x 3,310 x 2,499 and therefore does not explain the paper's
dimensions.

Example:

```sh
./standalone/prepare_data \
  --input data/raw/hetrec2011-lastfm-2k/user_taggedartists.dat \
  --output data/processed/lastfm/triples.csv \
  --min-count 5
```

DBLP is deliberately excluded from this reproduction at the user's request.
