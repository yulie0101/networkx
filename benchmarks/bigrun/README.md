# bigrun -- portable large-scale benchmark for max_cardinality_matching_gabow

A self-contained package to run this project's correctness and complexity
experiments at a much larger scale than a typical dev laptop, on a machine
with more RAM and CPU cores. No C++ and no Claude Code required -- just
Python and internet access (to install the package from GitHub).

## How to run

Copy this whole `bigrun` folder to the other machine, then double-click
`run.bat`, or from a command prompt in this folder:

```
run.bat
```

That one command:

1. Checks Python is installed (prints install instructions if not).
2. Creates a virtual environment (`venv\`) if one doesn't already exist.
3. Installs `networkx` from this project's branch (`git+https://github.com/
yulie0101/networkx@add-gabow-maximum-cardinality-matching`), plus
   `matplotlib` and `psutil`.
4. Runs `bigrun.py` (the experiments) and then `make_plots.py` (the plots).

## Expected runtime

Depends heavily on the machine and `config.json`'s sizes. The default
config includes Gabow runs up to ~1,000,000 nodes (fast) and Edmonds
comparisons up to `edmonds_max_n` = 45,000 vertices (Edmonds is O(n^3)).
Each graph's Edmonds time is the median of `edmonds_repeats` = 3 separate
calls, each capped at `edmonds_timeout_sec` = 1800s, so the speedups at
n >= 20,000 are not single samples; Gabow still runs at every size. The
Edmonds repeats at n = 20,000-40,000 dominate the runtime: on a machine
with 8+ cores and 32 GB RAM, expect several hours for the full default
config. You do not need to babysit it.

## Resuming an interrupted run

Every finished run is appended to its CSV immediately. If `run.bat` is
interrupted (closed, machine restarted, etc.), just run it again -- it
reuses the existing `venv\` and `bigrun.py` skips every (family, n, seed,
...) combination already present in `results\bigrun_correctness.csv` /
`results\bigrun_complexity.csv`. Nothing is redone.

To check progress without running anything:

```
venv\Scripts\python bigrun.py --report
```

## Where results go

- `results\bigrun_correctness.csv` -- one row per (n, density, seed,
  order, greedy): validity, Gabow size, Edmonds reference size (if it
  didn't time out), pass/fail.
- `results\bigrun_complexity.csv` -- one row per (family, n, seed,
  algorithm, variant): wall-clock time, iteration count, operation
  counts. Edmonds rows hold the median of their repeats
  (`edmonds_repeats`, with `runtime_min`/`runtime_max`).
- `results\environment.json` -- CPU, RAM, Python version, networkx
  commit, for reporting alongside the numbers above.
- `results\plots\P1..P4*.png` -- iterations vs n, ops/bound, wall-clock
  per family, speedup vs n (all families, Edmonds/Gabow, y=1 line).

## Changing the experiment sizes

Edit `config.json` (sizes, seeds, densities, `edmonds_timeout_sec`,
`max_workers`). No code changes needed. A tiny config for a quick local
test might look like:

```json
{
  "correctness": {
    "n_small": [50],
    "n_large": [200],
    "seeds": [0],
    "orders": ["ascending"],
    "greedy": [true]
  },
  "complexity": {
    "F1_sparse": { "c_values": [3], "sizes": [100], "seeds": [0] },
    "F2_dense": { "densities": [0.5], "sizes": [50], "seeds": [0] },
    "F3_bad_greedy": { "sizes": [20], "seeds": [0] },
    "F4_nested_blossoms": { "sizes": [11], "seeds": [0] },
    "F5_chains": { "n_params": [100], "seeds": [0] },
    "F6_long_path": { "sizes": [100], "seeds": [0] }
  },
  "edmonds_timeout_sec": 10,
  "edmonds_repeats": 3,
  "max_workers": 2
}
```

Run with `venv\Scripts\python bigrun.py --config my_tiny_config.json`.
