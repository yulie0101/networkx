# Change review: `max_cardinality_matching_gabow`

This appendix lists every change made to the algorithm code
(`networkx/algorithms/matching.py`) on the branch
`add-gabow-maximum-cardinality-matching`, the differences that remain
between our port and the C++ reference it was ported from, the bugs found
along the way, and what has and has not been verified.

"The C++ reference" means `GabowBeautified.h` from Mehlhorn's companion
page for Mehlhorn & Nobahari (arXiv:2603.22909). The file we were
originally given (`max_matching_c_2plus.cpp`) is content-identical to it
(see `docs/cpp_provenance.md`). `GabowRevised.h` is a different, later
file from the same page and is mentioned only for comparison.

## 1. Changes to the algorithm code, in order

| Commit                             | Date       | What changed                                                                                                                                                                                                                                                                                                | Why                                                                                                                                                                 | Effect on correctness                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Effect on complexity                                                                                                                                                         |
| ---------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `6cee35fa4`                        | 2026-09-17 | First version of `max_cardinality_matching_gabow`: Phase 1 (Delta-bucket search, blossom shrinking), Phase 2 (depth-first search on the contracted graph H), greedy warm start, iterative (non-recursive) helpers, union-find with path compression. Also had an opt-in `use_heuristic_fallback` parameter. | Port of the C++ reference into NetworkX.                                                                                                                            | Baseline. It contained two latent defects found later: the LIFO bucket order (bug 2 below) and a quadratic list insert that also reversed Fig. 4's exploration order (performance defect, section 3).                                                                                                                                                                                                                                                                                                  | Intended O(√n·m) per the paper; actually O(√n·m·log n) because the union-find had no union by size, plus the quadratic insert in one loop.                                   |
| `ddfa4edcb` (author: MayaGouldman) | 2026-10-05 | Phase 2 blossom-absorption loop: `tmp.insert(0, cur)` changed to `tmp.append(cur)`.                                                                                                                                                                                                                         | `insert(0, …)` shifts the whole list each time.                                                                                                                     | None on the matching size (636 stress-graph trials at the time). It also restores the exploration order of Gabow's Fig. 4 (u_1 first); the old code explored the absorbed vertices in reverse. See section 3 for what the reversed order could affect.                                                                                                                                                                                                                                                 | Removes an O(K²) cost per blossom step (K = vertices absorbed in that step), which could exceed the O(m) per-iteration budget on clique-like graphs. Now O(K).               |
| `3806b99b5`                        | 2026-10-05 | Added the internal instrumentation parameters `_counters` (operation counts per iteration) and `_skip_greedy_init` (start from an empty matching, as in the paper's Fig. 1).                                                                                                                                | Needed to measure iterations and work for the complexity experiments.                                                                                               | None: counters only increment; the default path is unchanged.                                                                                                                                                                                                                                                                                                                                                                                                                                          | One extra `is not None` check per instrumented line; no change in order of growth.                                                                                           |
| `b0c76ef47`                        | 2026-10-05 | Union-find now uses union by size as well as path compression. A separate `rep` map keeps the blossom base as the reported representative. The old behaviour is still available with `_union_by_size=False`.                                                                                                | Path compression alone gives O(log n) per operation; with union by size it is O(α(n)), matching the C++ reference's LEDA union-find.                                | None: identical matchings on 300/300 random graphs across old, new and compression-only versions; full test suite passed.                                                                                                                                                                                                                                                                                                                                                                              | Per-iteration cost goes from O(m·log n) to O(m·α(n)), so the total is O(√n·m·α(n)).                                                                                          |
| `8927e3d4d`                        | 2026-10-05 | Merge bringing in `ddfa4edcb`; comment next to `tmp.append` explaining the order difference from the C++ `push_front`.                                                                                                                                                                                      | Merge of teammate's test and visualization work.                                                                                                                    | None (comment only in the algorithm).                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | None.                                                                                                                                                                        |
| `8bc27922b`                        | 2026-10-07 | Added the internal parameter `_initial_mate`: start from a given matching instead of the greedy or empty start.                                                                                                                                                                                             | To test the C++ reference's own starting matching (edge-scan greedy) on the F5 chain graphs; this later explained the gap to the paper's Table 1.                   | None: unused by default; the caller must pass a valid matching.                                                                                                                                                                                                                                                                                                                                                                                                                                        | None.                                                                                                                                                                        |
| `d2411bfc4`                        | 2026-10-07 | Phase 1: a popped bucket entry with both ends EVEN is now rechecked for tightness before it is treated as a blossom or an augmenting path. Added regression tests.                                                                                                                                          | Fix for bug 2 (wrong, too-small matching on 52 of 120,000 random runs).                                                                                             | Fixes bug 2: 52 failures before, 0 after on the same 120,000 runs. **The commit message is imprecise:** it calls the bug a "missing tightness check vs C++ reference" and names `GabowRevised.h` as the file we ported from. The commit predates the Task B investigation, which showed that the real port source, `GabowBeautified.h`, has no such check either; the actual root cause was the LIFO/FIFO bucket order (see `d32e26a7d` and bug 2). The recheck is correct and is kept as a safeguard. | None in order of growth: the check is O(1) beyond the `find` calls already made for each popped entry, and it never re-inserts entries. Measured cost about 2–3% wall-clock. |
| `94fd13bdc`                        | 2026-10-07 | Removed `use_heuristic_fallback` and the code that existed only for it.                                                                                                                                                                                                                                     | The fallback is not part of Gabow's algorithm, behaved differently from the C++ driver's own heuristic, and, when on, breaks the O(√n) iteration bound.             | None on the default path: identical matchings on more than 2,000 graphs.                                                                                                                                                                                                                                                                                                                                                                                                                               | None on the default path: every `_counters` value identical before and after on 9 graphs (F1, F2, F5).                                                                       |
| `d32e26a7d`                        | 2026-10-08 | Delta-bucket queue changed from LIFO (`list.pop()`) to FIFO (`deque.popleft()`), as in the C++ reference (LEDA `list::pop` removes the first element). Tightness recheck kept.                                                                                                                              | This is the real root cause of bug 2, found by Task B (see below).                                                                                                  | Restores the reference's processing order. Matching sizes identical on 2,452 graphs, including all 52 known cases.                                                                                                                                                                                                                                                                                                                                                                                     | None: insert and pop are both still O(1). Counters shift by under 5% (a different but equally short augmenting path is found); F5 iteration counts unchanged.                |
| `995308a86`                        | 2026-10-08 | If Phase 1 finds an augmenting path but Phase 2 finds none, the function now raises `NetworkXAlgorithmError` instead of silently stopping. Comments and docstring rewritten (paper references, union-find wording, list of differences from the C++ reference).                                             | A silent stop would return a matching that is not maximum with no warning. By the paper's Corollary 3.3 this case is impossible, so if it ever happens it is a bug. | Turns a possible silent wrong answer into an error. Never triggered in any test or experiment.                                                                                                                                                                                                                                                                                                                                                                                                         | None.                                                                                                                                                                        |

Commits that changed only tests: `4b7580b7d` (2026-09-23, extra tests
and visualizations) and `e4c20aea2` (2026-10-07, restored 27 tests that
had been removed together with `use_heuristic_fallback`). Benchmark-only
commits (`benchmarks/`) are not listed.

## 2. Remaining differences from `GabowBeautified.h`

None of these changes the size of the returned matching.

1. **Tightness recheck on EVEN–EVEN entries (kept).** The reference has
   no recheck. Ours rechecks before treating an entry as a blossom or
   augmenting path. Reason: it was needed with our old LIFO queue, it
   costs almost nothing, and nobody has proved it is unnecessary with
   FIFO (see section 4). The later `GabowRevised.h` has an equivalent
   check.
2. **Order of two lists (tie-breaks only).** We append to `T` where the
   reference prepends, and Phase 2 picks its starting roots in a slightly
   different order. Reason: appending is O(1) in Python, while prepending
   to a list is not. Effect: a cross-check against the compiled reference,
   using a scratch copy of our code with these two orders changed to the
   reference's, found **all 6,320 cases identical edge for edge** (same
   matching, number of iterations and augmentations per iteration).

   The Phase 2 list `tmp` is **not** a difference, although it looks like
   one: we append where the reference prepends (`push_front`), but the
   reference explores `tmp` by recursion and we use a LIFO stack, so both
   explore u_1 first (see the performance-defect entry in section 3).
   An earlier version of that cross-check also changed `tmp` to prepend,
   which reversed the order, and reported 6,314 of 6,320 identical; with
   `tmp` left as in our code, the 6 remaining differences disappeared.
   (`cpp_provenance.md` §2.3 and the Notes in `matching.py`'s docstring
   still describe the older result.)

3. **Dual values reset every iteration.** We reset `bd`/`bDelta` to 1/0
   for every vertex at the start of each iteration, following the ADM24
   paper ("we initialize d(v) to one for all v"). The reference keeps
   them between iterations. Checked earlier with the old cross-check:
   changing ours to match did not change any result. Since the
   cross-check is now identical on all 6,320 cases without this change,
   the reset makes no observable difference there.
4. **Error instead of silent stop when an iteration gains nothing.** The
   reference has no such check (its loop would repeat forever). See
   `995308a86` above.
5. **Explicit stacks instead of recursion.** The reference uses C++
   recursion in the Phase 2 search and the two path-reconstruction
   routines. Python's recursion limit is far too small for long paths and
   cycles, so we use explicit stacks, pushing children in reverse order
   to keep the same depth-first order.
6. **No heuristic fallback.** The reference's timing driver switches to
   a one-path-at-a-time heuristic near the end. It is not part of
   Gabow's algorithm and breaks the O(√n) iteration bound, so we do not
   offer it.
7. **Different greedy warm start.** Our default start is a vertex-scan
   greedy (for each free vertex, take its first free neighbour); the
   reference's `init()` is an edge-scan greedy (scan edges in insertion
   order). They can give very different starting matchings on the same
   graph. The reference's version is available for experiments through
   `_initial_mate`.
8. **Fresh per-iteration structures.** We rebuild the labels, dual
   values, union-find and bucket array at the start of every iteration
   (O(n) each time); the reference reuses its arrays. This adds an
   O(n·√n) term, which is why the bound is stated as O(√n·(n+m)·α(n))
   for graphs with isolated vertices.

## 3. Bugs found

### Performance defect (not a correctness bug): quadratic list insert in Phase 2, whose fix also restored the order of Gabow's Fig. 4

Results were correct before and after this change; it is listed here
because it was a real defect in the port.

- **How found:** changed by Maya Gouldman in `ddfa4edcb` (commit
  message: "Fix potential matching issue"); its cost was analysed
  afterwards in the static complexity audit
  (`docs/complexity_audit.md` §2c). Note on attribution: the merge
  commit `8927e3d4d` calls this "Shai's" change by mistake, and the
  backup branch name `backup/pre-merge-shai` repeats the same mistake.
  The change is Maya's, as the commit's author field shows.
- **Root cause (cost):** `tmp.insert(0, cur)` in the blossom-absorption
  loop costs time proportional to the list length, so building a list of
  K vertices costs O(K²).
- **Root cause (order):** a blossom step in Phase 2 makes the odd
  vertices u_1, …, u_k on the tree path even again (u_1 is closest to
  the base b(x)). Gabow's Fig. 4, line 6, explores them in that order:
  "for i ← 1 to k do find_ap(u_i) /\* process u_i in order of increasing
  depth \*/". The C++ reference does this with `tmp.push_front` plus
  recursion. Our port replaces the recursion with an explicit LIFO
  stack, which reverses push order: with `append`, u_1 ends on top and is
  explored first, as in Fig. 4. The old `insert(0, …)` put u_k on top, so
  the old code explored them in the **reverse** order. Run traces
  confirm this (3,000 random graphs, greedy on and off, blossom steps
  with at least 2 absorbed vertices): current code u_1 first in 114/114,
  old code u_k first in 114/114, compiled C++ reference u_1 first in
  55/55.
- **Fix:** `tmp.append(cur)` (`ddfa4edcb`), which fixed both the cost
  (O(K²) → O(K)) and the order.
- **Guarding test:** none specific to the cost or the order; the blossom
  tests (nested, chained, windmill and random graphs checked against
  `max_weight_matching`) cover the results.
- **What the reversed order could affect, in theory.** The order matters
  for the proof that each iteration finds a _maximal_ set of
  vertex-disjoint shortest augmenting paths, not for whether the
  returned matching is maximum:
  - Gabow (2017), Sec. 4, on property (P1) ("every outer vertex that has
    not been completely scanned is on a path of P"): "Line 6 is important
    for this." His proof of (P1) in Appendix A uses the order directly:
    "When find_ap(u_1) returns, u_1 is completely scanned, so induction
    (applied to u_1) shows P(u_2) contains every new outer vertex not
    completely scanned. … This pattern continues for u_3, …, u_k."
    (P1) and (P2) together give maximality.
  - Mehlhorn & Nobahari, Sec. 3.2: "we should put u*k to u_1 (in this
    order!!!) on the recursion stack … and start to explore edges out of
    u_1 … If the calls were made in a different order, the above would
    not be true. For example, if we were to search from u_k first and be
    successful, the removal of the augmenting path would remove the
    justification for the nodes u_1 to u*{k−1} to be even."

  So with the reversed order the proof of maximality no longer applies.
  Correctness of the final result does not depend on it: every path
  found is still a valid augmenting path (it is rebuilt from parent and
  bridge pointers, not from the stack), and the outer loop continues
  until Phase 1 finds no augmenting path. What is at risk is the
  O(√n) bound on the number of iterations, which needs maximal sets.

- **What it did in practice.** Scratch experiment: the current code
  against the same code with only `tmp.insert(0, cur)` restored, on
  4,000 random graphs (8–60 vertices, greedy on and off) plus F3, F4 and
  F5 (edge-scan start) instances, 8,010 runs each.
  - Final matching: valid and of maximum size in all 8,010 runs of both
    versions.
  - Hopcroft–Karp invariant (the shortest augmenting path length strictly
    increases from one iteration to the next; a violation would mean an
    iteration left a same-length path behind, i.e. its set was not
    maximal): 0 violations in either version.
  - Iterations: equal in 8,008 runs; the old order needed one more
    iteration in 2 runs (one 50-vertex graph, greedy on and off) and never
    fewer. In the length-5 iteration of that graph the current code
    augments 2 disjoint shortest paths, the old order finds 1 whose
    augmentation destroys the other, which then has to be found later as
    a length-7 path. The old order's set was still maximal, just smaller.

  Conclusion: the old order was more than a tie-break in theory, because
  the published maximality proof does not cover it, but no harm to
  correctness was possible and none to maximality was observed; its only
  observed effect was an occasional extra iteration.

### Bug 2: stale bucket entries gave a matching one edge too small

- **How found:** an independent review ran 20,000 random graphs
  (n = 6–40, several densities, 3 node orders, greedy on and off;
  120,000 runs in total) against `max_weight_matching` and found 52 runs
  where our matching was one edge smaller than the maximum. Smallest
  example: a 7-cycle 0-2-7-12-4-1-13-0 with a path 1-5-8-16 attached,
  nodes inserted in ascending order: 4 edges returned instead of 5.
- **Root cause:** the Delta-bucket queue was LIFO in our port and FIFO
  in the reference. The reference's queue uses LEDA's `list::pop`, which
  removes the first element; the port translated it to Python's
  `list.pop()`, which removes the last. With LIFO, an entry inserted with
  a predicted pop time can be popped after one of its endpoints has
  changed state (unlabeled → odd → even), so it is no longer tight but is
  still treated as an augmenting path. Phase 2 then finds nothing, and the
  old code stopped one augmentation early.
- **How the root cause was confirmed (Task B):** the first fix
  (`d2411bfc4`) added a tightness recheck and blamed a missing check. A
  later investigation compiled the reference itself and found:
  - **No recheck in the reference either.** The reference has no such
    check.
  - **The reference never fails.** Run on the same inputs, it failed on
    **0 of 232,195** cases, including all 52.
  - **FIFO alone fixes our old code.** A FIFO-only patch of our old code,
    without the recheck, also passed all 232,195.
  - **LIFO alone breaks the reference.** Forcing only the reference's
    bucket queue to LIFO reproduced the bug exactly, including on the
    10-node example.
- **Fix:** FIFO order restored (`d32e26a7d`); the recheck from
  `d2411bfc4` is kept as a safeguard.
- **Guarding tests:**
  - `test_stale_bucket_entry_minimal_example` (the 10-node graph) and
    `test_stale_bucket_entry_randomized` (72 cases: 12 random graphs ×
    3 node orders × greedy on/off, each checked against
    `max_weight_matching`). These guard the result, but pass with either
    fix alone.
  - `test_fifo_bucket_order_no_stale_entries_known_cases` (all 52 known
    cases) pins the FIFO order itself: it asserts that no popped entry
    fails the tightness recheck (`stale_bucket_entries_skipped == 0`).
    Checked on a scratch copy of the code with only `popleft()` changed
    back to `pop()`: all 52 cases fail (1 or 2 stale entries each), and
    they pass on the current code.
  - A companion test on 32 random graphs
    (`test_fifo_bucket_order_no_stale_entries_random`) checks the same
    thing more broadly. It passes on both versions, so it does not by
    itself detect LIFO; the 52 known cases do.

### Bug 3: silent stop when an iteration gains nothing

- **How found:** while analysing bug 2: the old code ended the search
  quietly when Phase 2 found no path, which is how bug 2 turned into a
  wrong answer instead of an error.
- **Root cause:** a defensive `break` in the outer loop.
- **Fix:** raise `NetworkXAlgorithmError` (`995308a86`).
- **Guarding test:** none can trigger it, since the situation is
  impossible when the algorithm is correct; every correctness test
  implicitly checks it never fires.

### Not a bug in the algorithm: benchmark issues found along the way

These were in the benchmark scripts, not in `matching.py`, and were
fixed there:

- **Edge order for the C++ start.** The F5 "edge-scan" start was built
  from `G.edges()`, which is not the C++ insertion order, so it solved
  the family in one iteration.
- **Shared Edmonds cut-off.** The Edmonds time limit was shared between
  different F1 densities (c values).
- **Colliding resume keys.** In the `bigrun` package the resume keys
  ignored c and density, so different families overwrote each other.

## 4. What was verified and what remains open

### Verified

- **Test suite:** 445 tests in `test_matching.py` pass (374 of them for
  this function), including named graphs with known matching size,
  comparison with independent algorithms (Hopcroft–Karp on bipartite
  graphs, an exact tree algorithm), nested and chained blossoms, and
  inputs too deep for recursion.
- **FIFO order is pinned by a test.** On the 52 known stale-entry cases
  the current code never skips a stale entry, and a copy of the code
  with only the LIFO order restored fails all 52 (see bug 2).
- **Random correctness against `max_weight_matching`:**
  - 120,000 runs on small graphs (0 failures after the fix).
  - 3,024 runs on larger graphs, 0 failures: 200–1,000 vertices at
    three densities (m = 3n, p = 0.1, p = 0.5) and 1,200–3,000 vertices
    sparse (m = 3n), each with 3 node orders and greedy on/off.
- **Against the compiled C++ reference:**
  - The 232,195-case sweep: the reference gave the correct size on every
    case. A FIFO-only patch of our old code also passed every case, and
    our current code passed the 2,452-graph check.
  - The 6,320-case same-order cross-check (52 known cases, a 6,000-graph
    battery, small F3/F4/F5 instances; two starting matchings each):
    **6,320 of 6,320 identical** edge for edge, in iteration count and in
    augmentations per iteration, once the `T` and root orders are matched
    and `tmp` is left as in our code (section 2, item 2).
- **Complexity, by reading the code:** a line-by-line audit
  (`docs/complexity_audit.md`) found O(m·α(n)) work per iteration plus
  the O(n) re-initialization, and no recursion.
- **Complexity, by measurement:** on the C++ start for the F5 chain
  graphs, the iteration count grows like n^0.52 (n = 5,026–80,180
  vertices) and total work divided by √n·m stays flat at about 3.2. The
  O(√n) iteration bound is therefore reached, up to a constant factor, on
  this family.
- **Paper's Table 1 explained:** simulating the C++ driver's heuristic
  trigger on our runs gives exactly 24, 33 and 47 iterations, the values
  in Table 1.

### Open

- **Is the recheck needed with FIFO?** No failure was found without it
  in 232,195 cases, but neither the Mehlhorn & Nobahari paper nor ADM24
  states an invariant that rules out a stale entry under FIFO. This is
  empirical evidence, not a proof; the recheck is kept.
- **Per-iteration re-initialization.** The O(n) rebuild in every
  iteration is disclosed but not removed. Removing it would need a
  larger refactor.
- **Blossom-absorption cost.** The amortized cost of walking up and
  absorbing blossoms rests on the standard argument re-derived from the
  code. No experiment measures it directly.
- **Untested options.** `_union_by_size=False` and `_initial_mate` have
  no dedicated tests (they are internal and used only in benchmarks).
- **Unverified LEDA default.** The value LEDA uses for an untouched
  `node_array<int>` (relevant to difference 3) was taken from our
  compiled stand-in, not from real LEDA.
- **Smallest Table 1 row.** The paper's smallest short-and-long-chains
  instance (n = 10,000, m = 22,000) has fewer edges than our graph of the
  same vertex count (28,388). The other rows match within 2%.
- **Dense graphs with an empty start.** With an empty starting matching
  (greedy off), our implementation is slower than NetworkX's Edmonds
  implementation on dense random graphs at n = 800: about 1.1 times at
  10% density and 2.2–3.4 times at 25–100% density. This is about
  constant factors in Python, not about the asymptotic bound.
- **Greedy-start speedups on dense graphs depend on how runs are
  aggregated.** With the default greedy start, running times are bimodal:
  on some graphs the greedy matching is already maximum and Gabow
  finishes in milliseconds, on others it runs the full search. Example,
  n = 400, 75% density, same `--full` run as P6/P9 (5 graphs):
  - _Per-graph medians:_ Edmonds/Gabow = 0.72×, 1.09×, 0.84×, 31× and
    42×. Greedy was already maximum on the two fast graphs; Gabow is
    slower on 2 of 5 graphs, and the median over graphs is 0.84×.
  - _Median over all timed runs_ (what P6/P9 plot): about 30×. Fast calls
    are repeated until 2 s have elapsed, so the two fast graphs
    contribute 100 of the 133 Gabow timings and dominate this median.

  Both numbers come from the same measurements. P6/P9 should be read as
  "typical run", not "typical graph"; per-graph medians would be the
  fairer summary for bimodal points.
