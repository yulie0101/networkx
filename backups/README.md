# Frozen backup: `gabow-v1-backup`

Snapshot of the Gabow maximum-cardinality-matching implementation, its
tests, and its benchmark/verification scripts, taken at git tag
`gabow-v1-backup`, before adding union-by-size to `_UnionFind` and before
investigating the short-and-long-chains iteration-count discrepancy.

**Do not edit these files.** They exist so the current implementation can
be compared side by side against this known-good prior version. All
further changes go into `networkx/algorithms/matching.py` and the live
`benchmarks/`/`networkx/algorithms/tests/` files, not here.

- `matching_gabow_v1.py` — standalone, importable copy of
  `max_cardinality_matching_gabow`, renamed to
  `max_cardinality_matching_gabow_v1` so both versions can be imported in
  the same script without clashing.
- `matching_tests_v1_snapshot.py` — snapshot of
  `networkx/algorithms/tests/test_matching.py` at the same point. Renamed
  from `test_matching_v1.py` (no `test_` prefix) specifically so pytest's
  default discovery does not collect it alongside the live test suite.
- `bench_matching_tests_v1_snapshot.py` — same, for
  `networkx/algorithms/tests/test_bench_matching.py`.
- `bench_matching_v1.py`, `verify_complexity_v1.py` — snapshots of the
  corresponding `benchmarks/` scripts.
