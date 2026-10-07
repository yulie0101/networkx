# Static complexity audit — `max_cardinality_matching_gabow`

**Scope.** Stage A only: static, from-the-source-code inspection of the
_current, merged_ `max_cardinality_matching_gabow` in
`networkx/algorithms/matching.py` (as of commit `8927e3d4d`, lines
1156–1891). No operation counts, benchmarks, or plots were run to produce
this document — every claim below is either read directly off the code or
derived by a charging/amortization argument stated explicitly as such. No
algorithm logic was changed while producing this report.

Terminology follows the function's own docstring (Mehlhorn & Nobahari's
convention): one **iteration** = one Phase 1 + Phase 2 round (the outer
`while True:` loop body, line 1867); one **Δ-phase** = one value of `Delta`
visited inside Phase 1's `while 2 * Delta <= n:` loop (line 1581).

---

## 1–2. Loop-by-loop audit table

Costs below are **per iteration** unless marked "one-time" (outside the
iteration loop entirely) or "amortized total" (bounded only when summed
over every Δ-phase/loop-invocation _within_ one iteration, not per single
invocation — flagged explicitly where the per-invocation cost alone would
look worse than the true total).

| Lines           | Operation                                                                                                                                                                               | Cost per iteration                                                                                                                                                                                                                                                                                                                                                                      | OK / PROBLEM                                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1337–1348       | `n = G.number_of_nodes()`; `adj = {...}` dict comp over `G[v]`                                                                                                                          | One-time, O(n+m), outside the loop (runs once per _call_, not per iteration)                                                                                                                                                                                                                                                                                                            | OK                                                                       |
| 1354–1365       | Greedy warm-start: `for v in G: ... for u in adj[v]: if u not in mate:`                                                                                                                 | One-time, O(n+m): each vertex's adjacency scanned until a match or exhaustion, total bounded by `sum(deg(v))` = O(m)                                                                                                                                                                                                                                                                    | OK                                                                       |
| 1427–1435       | `_UnionFind.__init__`: `list(elements)`, `dict.fromkeys(...)` ×2, `dict(self.parent)`                                                                                                   | O(n) each of two `_UnionFind` instances (`base`, `dbase`) built fresh **every iteration**                                                                                                                                                                                                                                                                                               | OK, but see §3 (re-init total)                                           |
| 1484–1486       | `_BucketQueue.__init__`: `[[] for _ in range(size)]`, `size = n//2+1`                                                                                                                   | O(n) per iteration                                                                                                                                                                                                                                                                                                                                                                      | OK, but see §3                                                           |
| 1507, 1512–1513 | `label = {...}`, `bd = {...}`, `bDelta = {...}` dict comps over `G`                                                                                                                     | O(n) each, per iteration                                                                                                                                                                                                                                                                                                                                                                | OK, but see §3                                                           |
| 1527            | `T = [v for v in G if v not in mate]`                                                                                                                                                   | O(n) per iteration (initial build; `T` is then only ever `.append`ed to, never rebuilt mid-iteration)                                                                                                                                                                                                                                                                                   | OK                                                                       |
| 1575–1577       | Initial scan: `for v in T: for u in adj[v]: scan_edge(u, v)`                                                                                                                            | Each `v` here is currently-free; sum of `deg(v)` over free `v` ≤ O(m)                                                                                                                                                                                                                                                                                                                   | OK                                                                       |
| 1539–1550       | `scan_edge(u, z)`: `_counters` guard, `mate.get`, `base.find`, `d(z)+d(u)`, one `queue.insert`                                                                                          | O(α(n)) per call (dominated by the two `base.find` calls inside `d()`); **total calls per iteration**: each directed edge-endpoint is scanned at most twice overall (paper's own claim, Sec. 4 — "any edge is scanned at most twice, once from each end"), so total calls = O(m)                                                                                                        | OK — total O(m·α(n))                                                     |
| 1584–1636       | Δ-bucket dequeue loop: `while True: e = queue.pop(Delta); ...`                                                                                                                          | `queue.pop` is O(1) (`list.pop()`, no index — pops from the end); **total dequeues across all Δ in one iteration** ≤ total insertions ≤ O(m) (each `scan_edge` call inserts at most once)                                                                                                                                                                                               | OK                                                                       |
| 1596–1608       | Grow step (new ODD/EVEN pair, `T.append` ×2, inner `for u in adj[z]: scan_edge(u,z)`)                                                                                                   | `T.append` is amortized O(1); the inner scan is counted inside the O(m) total above, not double-counted since `z` becomes EVEN at most once                                                                                                                                                                                                                                             | OK                                                                       |
| 1610–1628       | **Ancestor-climb loop** (EVEN–EVEN case): `while (path1.get(hy)!=strue and path2.get(hx)!=strue) and (...)`, climbing via `base.find(parent[mate[hx]])`                                 | **Per single invocation**, worst case O(depth of current tree) — _could_ look like O(n). **Total across the whole iteration**: amortized O(n), via a charging argument (see §2a below) — NOT O(m) per invocation × O(n) iterations                                                                                                                                                      | OK, but requires the amortized argument below, not a naive per-line read |
| 1552–1573       | `shrink_path(b, x, y, dunions)`: `while v != b: base.union; dunions.append ×2; base.union; dunions.append ×2; base.make_rep; for u in adj[v]: scan_edge(u,v); v = base.find(parent[v])` | Same charging argument as above: each vertex absorbed into a blossom is visited by `shrink_path`'s while-loop **at most once per iteration** (once absorbed, `base.find` resolves it in O(α(n)) from then on) — total O(n) loop-steps, each doing O(deg(v)) work for the inner `scan_edge` fan-out, summing (via the same O(m) edge-scan budget above) to the already-counted O(m·α(n)) | OK                                                                       |
| 1639–1648       | `dunions` commit loop: `while i < len(dunions): ...; i += 2`; `dunions.clear()`                                                                                                         | `dunions[i]` is O(1) list indexing (not `.pop(0)`); total length of `dunions` across one iteration is O(number of vertices absorbed into blossoms) = O(n) (same charge as above); `.clear()` is O(len)                                                                                                                                                                                  | OK                                                                       |
| 1655–1673       | Build H: `rep = {v: dbase.find(v) for v in T}`; `for u in T: for v in adj[u]: ...`                                                                                                      | `rep` comprehension: O(\|T\|·α(n)) ≤ O(n·α(n)). Double loop: `sum(deg(u))` over `u in T` ≤ O(m) (plain iteration, no list lookups)                                                                                                                                                                                                                                                      | OK                                                                       |
| 1714–1718       | `edge_iter_for(vh)` generator: `for v in contracted_into[vh]: for u in adj[v]: if (v,u) in is_edge_of_H: yield`                                                                         | **Looks like neighbors-of-neighbors.** Bounded by the generator/stack design: each `vh` gets exactly one generator, created and (incrementally) fully consumed at most once per iteration (gated by `labelHG.get(uh) is None` before any push, line 1806/1819); summed over all `vh` ever visited, total work = `sum(deg(v))` over `v` partitioned across all visited `vh` ≤ O(m)       | OK, but easy to misread as quadratic — see §2b                           |
| 1728–1743       | `trace_HG`: explicit-stack loop, `stack.pop()`/`stack.append()` (O(1) each)                                                                                                             | Called once per augmenting path found this iteration; total work across all calls in one iteration ≤ O(size of H) = O(m) (vertex-disjoint paths, each H-edge touched O(1) times)                                                                                                                                                                                                        | OK                                                                       |
| 1752–1766       | `trace_G`: explicit-stack loop, same pattern                                                                                                                                            | Called from `augment()`, per found path; total ≤ O(n+m) by the same vertex-disjointness argument                                                                                                                                                                                                                                                                                        | OK                                                                       |
| 1768–1776       | `augment(h_edges)`: `for u,v in h_edges: mate_pairs.append(...); trace_G ×2`; final `for a,b in mate_pairs: mate[a]=b; mate[b]=a`                                                       | O(path length) per call; total across one iteration ≤ O(n)                                                                                                                                                                                                                                                                                                                              | OK                                                                       |
| 1779            | `for root in list(contracted_into):`                                                                                                                                                    | `list(contracted_into)` copies the dict's keys: O(\|contracted_into\|) ≤ O(n), **once per Phase 2 call**                                                                                                                                                                                                                                                                                | OK (bounded, not accumulating)                                           |
| 1788–1858       | Phase 2 main DFS `while stack:` loop, incl. blossom-absorption `while cur != bh:` (1831–1851)                                                                                           | `stack.pop()`/`.append()` O(1); the inner `while cur != bh:` loop has the **same amortized-O(n)-total charging argument** as Phase 1's blossom absorption (§2a) — each H-node absorbed into a bigger blossom is visited at most once per iteration, future lookups resolve via `dbase.find` in O(α(n))                                                                                  | OK                                                                       |
| 1848            | `tmp.append(cur)` (Maya's change, was `tmp.insert(0, cur)`)                                                                                                                             | **See §2c — this is the one genuine finding.**                                                                                                                                                                                                                                                                                                                                          | **Was PROBLEM, now fixed**                                               |
| 1813            | `found_h_edges = [(v, u)] + trace_HG(...)`                                                                                                                                              | `list + list` concatenation: O(1) + O(len(result)), happens **once** per found path, not inside a loop                                                                                                                                                                                                                                                                                  | OK (flagged per the task's checklist, not a real cost issue)             |
| 1868–1890       | Main iteration loop; `_snapshot`/`_diff` (only when `_counters` is not None)                                                                                                            | `_snapshot`/`_diff` build dicts over the ~14 fixed counter keys: O(1), independent of n/m                                                                                                                                                                                                                                                                                               | OK                                                                       |

### 2a. The ancestor-climb / blossom-absorption charging argument (the subtlest item)

Lines 1614–1622 (ancestor climb) and 1554–1571 / 1831–1851 (blossom
absorption) all walk up the _current_ blossom tree one matched-pair-hop at
a time. Read in isolation, a single invocation's cost is bounded only by
the current tree depth, which is not obviously O(1) or even O(√n) — a
naive reading would worry this could be O(n) per call, and since such
calls can happen up to O(m) times per iteration (once per dequeued
EVEN–EVEN edge), that naive reading would suggest O(n·m), which would
**break** the claimed bound entirely.

The actual bound relies on a **charging/amortization argument**, not a
per-call bound, and it is the same argument the Gabow (2017) paper itself
relies on (Sec. 3.1, the S̄ partition) and that the C++ reference this was
ported from implements identically:

- Every vertex visited by an ancestor climb or a `shrink_path` absorption
  is immediately afterward merged into the forming blossom (via
  `base.union`/`dbase.union` right there in the same loop body).
- Once merged, any _future_ reference to that vertex resolves through
  `base.find`/`dbase.find` in O(α(n)) (path compression + union by size),
  **never walking through it vertex-by-vertex again** within the same
  iteration.
- So each vertex can be the _target_ of a "walk through it one hop at a
  time" step **at most once per iteration** (the first time it's absorbed);
  every subsequent reference is an O(α(n)) `find`, already counted in the
  O(m·α(n)) edge-scan budget.
- Total ancestor-climb + absorption work per iteration is therefore
  O(n) amortized (bounded by the number of vertices, each charged once),
  not O(n) _per call_ — this additive O(n) term (already folded into the
  Verdict's O((n+m)·α(n)) per-iteration bound) is what keeps Phase 1/2 from
  degenerating into a multiplicative O(n·m).

This is a correct, standard argument for this _class_ of algorithm (the
same charging scheme used in Micali–Vazirani and Gabow–Tarjan), and the
code's structure (union immediately inside the climbing loop, before the
next step) is exactly what the argument requires. I verified the
structural precondition (union happens _inside_ the loop body, not after)
by reading the code directly; I have **not** independently reproduced a
formal proof of the amortized bound beyond restating the standard argument
— that would require the kind of empirical operation-counting this task
explicitly said to defer to a later stage.

### 2b. `edge_iter_for` — looks like neighbors-of-neighbors, isn't

`for v in contracted_into[vh]: for u in adj[v]:` (line 1715–1716) is a
nested loop over a blossom's member vertices and _their_ neighbors — the
exact shape the task asked me to watch for. It stays O(m) total (not
O(m·blossom size) or worse) only because:

1. `contracted_into` partitions `T` — every original vertex belongs to
   exactly one `vh`, so `sum(deg(v))` over all `v` across _all_ `vh`
   equals `sum(deg(v))` over `v in T` ≤ O(m), not O(m) _per_ `vh`.
2. The generator is created once per `vh` and is a genuine Python
   generator — `next(it, None)` resumes exactly where it left off; the
   `for u in adj[v]` loop is never restarted from the beginning for an
   already-partially-consumed `v`.
3. A `vh` is only ever pushed onto the DFS stack once (gated by
   `labelHG.get(uh) is None`, lines 1806/1819), so its generator is
   created at most once per iteration.

If any of those three held differently — if `contracted_into` had
overlapping membership, or the generator were re-created per visit instead
of resumed, or a `vh` could be revisited — this would silently become
O(m · average blossom count) instead of O(m). It currently does not.

### 2c. `tmp`: `insert(0, cur)` vs `append(cur)` — the one genuine finding

This is in the Phase 2 blossom-absorption loop (lines 1828–1848). `tmp`
collects the newly-absorbed H-nodes of one blossom step, one per pass of
the `while cur != bh:` loop (line 1831).

**Before Maya's change** (`tmp.insert(0, cur)`): `list.insert(0, x)` is
**O(current length of the list)**, because every existing element has to
shift one slot to make room at the front. Building a list of final size
_K_ via _K_ consecutive front-inserts costs `1 + 2 + ... + K` =
**O(K²)**, not O(K). _K_ here is the number of H-nodes absorbed in _one_
blossom step, which is not bounded by a constant — it can be as large as
the number of vertices in the blossom being formed (in last session's
experiments, a single clique-heavy iteration absorbed 70+ nodes into
blossoms). A blossom step absorbing a large fraction of `n` nodes in one
call would have cost O(n²) for just this `tmp`-building loop, which
**would exceed** the O(m)-per-iteration budget on graphs where a single
blossom is large relative to m (dense clique-like structures) — a real,
not merely theoretical, risk given the kinds of adversarial graphs this
investigation has specifically been using.

**After Maya's change** (`tmp.append(cur)`): `list.append` is O(1)
amortized, so building the same list of size _K_ costs O(K) total,
matching the O(n) amortized budget the rest of the blossom-absorption
loop already respects (§2a).

**Does anything later depend on `tmp`'s order?** Two consumers:
`for node in tmp: bridgeHG[node] = (v, u)` (order-independent — same value
written for every node) and `for node in tmp: stack.append(...)` (order
determines which absorbed node the DFS explores _first_, since the stack
is LIFO — `append` explores the last-absorbed node first, where
`insert(0, ...)` + the reference C++'s `push_front` would have explored
the first-absorbed node first). This changes _which_ valid maximum
matching is returned when multiple exist, not _whether_ one is found —
confirmed empirically in the prior merge investigation (636 stress-graph
trials, 0 mismatches between the two orderings), consistent with Phase
2's correctness argument (paper Appendix A, (P1)/(P2)) not depending on
which order newly-absorbed blossom members are explored in.

**Conclusion for this item:** this was a genuine latent O(K²) cost in one
specific loop, now fixed, and fixing it did not change (and by the DFS
non-determinism argument above, could not have changed) correctness —
only which optimal matching comes back and the loop's own cost.

---

## 3. Per-iteration O(n) re-initializations

| Structure              | Lines           | Rebuilt                                           | Total cost over all iterations |
| ---------------------- | --------------- | ------------------------------------------------- | ------------------------------ |
| `base` (`_UnionFind`)  | 1522, 1427–1435 | every iteration                                   | O(n) × (number of iterations)  |
| `dbase` (`_UnionFind`) | 1523, 1427–1435 | every iteration                                   | O(n) × (number of iterations)  |
| `_BucketQueue.buckets` | 1528, 1484–1486 | every iteration                                   | O(n) × (number of iterations)  |
| `label`                | 1507            | every iteration                                   | O(n) × (number of iterations)  |
| `bd`, `bDelta`         | 1512–1513       | every iteration                                   | O(n) × (number of iterations)  |
| `T`                    | 1527            | every iteration (rebuilt from `mate`, not reused) | O(n) × (number of iterations)  |

All six are rebuilt from scratch at the top of every `phase1()` call. This
is the **known, already-documented** divergence from the C++ reference,
which keeps a single `T` (and the label/dual-value bookkeeping tied to it)
as a **persistent member variable**, trimmed and regrown across
iterations rather than rebuilt — making the reference's equivalent cost
amortized O(n) **over the whole run**, not O(n) **per iteration**.

**Total cost of this divergence**: O(n) × (number of iterations). If the
number of iterations is O(√n) (the cited Hopcroft-Karp-style bound, not
verified from this code — see Verdict), this divergence contributes
O(n^1.5) additional total work. This is exactly why the Verdict below
states the per-iteration cost as O((n+m)·α(n)) rather than the
often-abbreviated O(m·α(n)): the explicit `+n` term already accounts for
this divergence in full (O(√n) · n = O(n^1.5) is one component of
O(√n·(n+m)·α(n))'s expansion), so it does not change the asymptotic class
of the overall bound on _any_ input, not merely ones with m = Ω(n) — it is
real, disclosed, un-amortized-across-iterations extra constant-factor work that
the reference avoids. **Not fixed in this task** (static audit only, no
logic changes).

---

## 4. Union-find audit

- Confirmed **union by size** (lines 1459–1465: compares `self.size[rx]` vs
  `self.size[ry]`, attaches the smaller under the larger) **plus path
  compression** (lines 1442–1449: both the walk-to-root loop and the
  separate compression-rewrite loop are present in `_find_root`).
- Confirmed **every** `base`/`dbase` lookup goes through `find()` — grepped
  all call sites of `.find(` in the function body; none bypass it to read
  `.parent`/`.rep` directly from outside the class.
- `make_rep` (lines 1469–1474): one `_find_root` call (O(α(n))) plus an
  O(1) dict write (`self.rep[root] = x`) and an O(1) counter increment.
  Adds O(1) beyond the `find` it already needed to do. Confirmed O(1)
  marginal cost per operation, as required.
- One structural note, not a defect: `union()`'s `_counters` increment
  (lines 1466–1467) is skipped when `rx == ry` (line 1457–1458, early
  `return`) — i.e., a no-op union (already in the same set) is not counted
  as a union. This matches intent (counting real merges only) but means
  the `uf_*_union_calls` counter undercounts _attempted_ unions relative to
  _successful_ ones; not a complexity issue, just a naming note for anyone
  reading the counters later.

## 5. Recursion audit

**No recursion anywhere in this function, direct or mutual.** Enumerated
every helper (`_snapshot`, `_diff`, `_UnionFind._find_root`/`find`/`union`/
`make_rep`, `phase1`, `d`, `scan_edge`, `shrink_path`, `phase2`,
`edge_iter_for`, `trace_HG`, `trace_G`, `augment`) and every call site
inside each one; the call graph is a strict DAG:

```
max_cardinality_matching_gabow
  -> phase1 -> scan_edge -> d, base.find
            -> shrink_path -> base.union, base.make_rep, scan_edge, base.find
  -> phase2 -> edge_iter_for (generator, calls nothing)
            -> trace_HG (explicit stack, calls nothing)
            -> augment -> trace_G (explicit stack, calls nothing)
            -> dbase.find/union/make_rep
```

No function appears twice on any path from the root, so there is no cycle
and hence no recursion (the explicit-stack pattern in `trace_HG`/`trace_G`
and the Phase 2 DFS loop is iterative _in place of_ what the C++
reference and the paper's pseudocode express recursively).

**Maximum Python call-stack depth added by this function**: a small
constant, independent of n and m — at most about 5 frames deep at any
point (e.g. `max_cardinality_matching_gabow -> phase1 -> shrink_path ->
scan_edge -> d`), never growing with graph size. This matches what was
already asserted in the project summary (last round) and in this
function's own docstring ("Every helper that the reference implements
recursively ... is implemented here with an explicit stack instead"); this
audit re-derives it directly from the current merged code rather than
relying on that earlier claim.

## 6. Section citations

Two comments in the current code cite specific paper sections:

- Line 1323 (docstring): `` `uf_base_*` for Sec. 3.1's S-bar partition ``
- Lines 1519–1521 (inline): `` `base` is Edmonds' S-bar blossom partition,
live during this search; `dbase` accumulates the *maximal positive*
blossoms (Sec. 3.4) and is committed to only once per Delta level. ``
- Line 1324 (docstring): `` `uf_dbase_*` for the maximal-positive-blossom
partition of Sec. 3.4 ``

**Sec. 3.1** matches the paper I was given (Gabow, 2017): Section 3.1,
"Edmonds' weighted matching algorithm," is exactly where the S̄ (S-bar)
structure — the alternating-forest-with-blossoms-contracted
representation — is defined. This citation is correct and verifiable
against the paper text.

**Sec. 3.4**: the copy of the paper I have shows Section 3 split into 3.1
/ 3.2 / 3.3 only ("Phase 1 via Edmonds' algorithm," with "Constructing
graph H" appearing as unnumbered material inside 3.3, not as a separate
3.4). I cannot verify "Section 3.4" as a real numbered section in _this_
version of the paper. This citation originates from the C++ reference
(`max_matching_c_2plus.cpp`, inherited verbatim: `// see section 3.4
Construction of H`), carried through into this port — it most likely
refers to a different numbering in an earlier draft/preprint the original
C++ author was working from, or possibly the LEDA documentation's own
section numbering, not necessarily a numbering mismatch introduced during
this port. The _content_ the comment describes (committing the `dunions`
pending-union list to `dbase` once per Δ-phase, which is what later lets H
be built from "maximal positive blossoms") is accurate and matches what
the paper's Sec. 3.3/Sec. 5 actually describe, independent of whether the
section number itself is right.

## 7. Other items checked

- **Dict/set O(1)-on-average, not worst-case**: `mate`, `label`, `bd`,
  `bDelta`, `parent`, `source_bridge`, `target_bridge`, `path1`, `path2`,
  `rep`, `contracted_into`, `mateHG`, `labelHG`, `parentHG`, `even_timeHG`,
  `bridgeHG` are all plain `dict`s; `is_edge_of_H` is a `set`. Every
  membership test / lookup against these (`v in mate`, `(v,u) in
is_edge_of_H`, `mate.get(u)`, etc.) is O(1) **expected**, not worst-case
  guaranteed, since Python dicts/sets are hash tables (vs. the C++
  reference's `node_array`/`edge_array`, true O(1)-worst-case arrays
  indexed by an internal id — already noted in last round's audit). Not a
  practical concern for this workload (int/tuple keys, no adversarial
  hashing exposure), but a real, disclosed difference in the strength of
  the guarantee.
- **No list used as a membership-tested collection anywhere.** The only
  `list`-typed structures in the function are `T`, `tmp`, `dunions`,
  `endpoints_of_M`, `mate_pairs`, `stack` (several), and `edges` (return
  value of `trace_HG`) — every one of them is only ever iterated over,
  appended to, indexed by position, or used as a LIFO stack (`.pop()`
  without an index, `.append()`). None is ever tested with `x in
some_list`.
- **No "neighbors of neighbors" nesting beyond `edge_iter_for`** (already
  covered in §2b) was found anywhere else in the function.
- **No `sorted()`, no `.index()`, no `.remove()`, no `pop(0)`, no other
  `insert(0, ...)`** anywhere in the current code (confirmed by direct
  grep over the full function body).

---

## 8. Addendum: correctness fix for stale Δ-bucket entries (post-Stage-A)

**Not part of the original Stage A audit** (commit `8927e3d4d`) — added
after an independent review found and reported a correctness bug: on a
small fraction of inputs (52/120,000 in a 20,000-graph search across
n=6–40, several densities, ascending/descending/random node insertion
order, greedy on/off), `max_cardinality_matching_gabow` returned a matching
one edge smaller than `max_weight_matching(maxcardinality=True)`'s.

**Root cause, as understood at the time of the fix.** `scan_edge` inserts
a bucket entry `(z, u)` at a _predicted_ future `Delta` computed from
`d(z)` and `d(u)` _at scan time_. If `u` is `UNLABELED` when scanned but
later transitions `UNLABELED` → `ODD` → `EVEN` (absorbed into a different
blossom) before that entry is popped, the prediction is stale: `d(u)` now
follows the `EVEN` formula instead of the `UNLABELED` constant the entry
assumed, so `d(x) + d(y) == 0` (tightness) no longer holds even though
`label[by] == "EVEN"` by the time it's popped. The code previously treated
every popped entry with `label[by] == "EVEN"` as a real
blossom-step/augmenting-path candidate with no recheck, so a stale entry
could cause a spurious augmenting-path detection, which (via the
`gained == 0` guard in the outer loop) stopped the algorithm one
augmentation early. The fix added a tightness recheck, `d(x) + d(y) == 0`,
gating the `EVEN` branch, citing `GabowRevised.h`'s equivalent
`lcp[x] + lcp[y] == 2 * Delta - 2` guard and calling the omission a
"translation miss" in the port.

**Superseded by Task B.** A later, dedicated investigation (Task B;
summarized here, full detail in `docs/cpp_provenance.md`) settled the
attribution differently, with decisive evidence: the actual port source,
`GabowBeautified.h` (confirmed content-identical to the originally-provided
`max_matching_c_2plus.cpp`), **does not have this recheck either** — its
EVEN branch is structurally identical to our pre-fix code, with no
tightness check anywhere. `GabowRevised.h` is a separate, later
reformulation (`lcp`/`lcp_odd` instead of `bd`/`bDelta`); its recheck is
not something our port's actual source ever had and omitted.

The real, confirmed translation error is that our `_BucketQueue` pops
LIFO (`self.buckets[d].pop()`, same end as `insert`'s `.append()`) where
the reference's `simple_queue` pops FIFO (LEDA's `list::pop()` removes the
**first** element; `insert` appends at the back). Dynamic evidence: with
the compiled, unmodified `GabowBeautified.h` (FIFO, as shipped) and our
own initial matchings/node orders, **0 of 232,195** test cases failed,
including all 52 originally-failing cases and a further 226,143-case
sweep (random graphs n=6–60, several densities, 3 node orders, 3 initial-
matching strategies, plus windmill/chained-triangle/nested-blossom/
bad-greedy families) run against a FIFO-only patch of the pre-fix Python
(recheck still absent). Forcing only the C++ bucket to LIFO (isolated from
every other list in the class) reproduced the original bug exactly,
including on the minimal 10-node example. A full faithful-order
cross-check (bucket FIFO, `T`/`tmp` prepended, Phase 2 root order matching
`T`'s scan order — see `docs/cpp_provenance.md` for the complete list, incl.
one previously-undocumented divergence found while building this
cross-check) against the compiled reference, with the recheck kept, found
**6,314 of 6,320 cases** (52 known + 6,000-graph battery + small F3/F4/F5
instances, two initial matchings each) identical edge-for-edge, with 100%
agreement on matching size, iteration count, and augmentations per
iteration in every single case; the remaining 6 differ only in _which_
same-size maximum matching comes back.

**Conclusion.** The recheck fix is correct, necessary given the port's
actual (LIFO) queue, and is kept — restoring FIFO order is a separate fix,
applied afterward (see the FIFO-fix commit and `docs/cpp_provenance.md`).
Whether the recheck is also necessary once the queue is FIFO remains
formally open (232,195 cases found no counterexample, but neither
Mehlhorn & Nobahari nor ADM24 state an invariant that rules one out for
this specific, `bd`/`bDelta`-based formulation — see
`docs/cpp_provenance.md`). The grow-step branch is provably unaffected
either way: `EVEN` is permanent for the rest of a search once assigned,
and `d(v) == 1` unconditionally while `v` stays `UNLABELED`, so neither
side of a still-`UNLABELED` entry can have drifted since it was scanned
(proved in the code comment at the fix site).

**Complexity impact: none, by construction.**

- _Cost per popped entry_: the added check is `d(x) + d(y) == 0`, i.e. two
  calls to `d(v)`, each one `base.find(v)` call plus O(1) arithmetic — the
  same O(α(n)) cost already paid one line earlier for `bx, by =
base.find(x), base.find(y)` on every popped entry, fix or no fix. No
  loop, no recursion, no new per-entry work beyond this.
- _Bucket insertions per iteration_: **unchanged**. The fix touches only
  the code that runs _after_ an entry is popped; it does not call
  `queue.insert` anywhere, does not change `scan_edge`'s call sites or
  logic, and never re-inserts a skipped entry. The O(m) bound on
  insertions per iteration (each edge scanned O(1) times per
  vertex-absorption event, §1–2 above) is exactly as before.
- _No new per-iteration structure_: no new array, dict, or counter sized
  beyond O(n + m) (`stale_bucket_entries_skipped` is a single `int`).
- If anything, the fix can only ever _reduce_ work in a given run: entries
  that previously, incorrectly, triggered `shrink_path` (O(blossom size))
  or a spurious `found_sap` now correctly do nothing instead.

The per-iteration bound therefore remains O(m·α(n)) exactly as audited in
§1–2, and the total bound remains O(√n · m · α(n)) (§8 does not change
anything about the √n iteration-count argument, which is independent of
this fix). See `benchmarks/run_all.py`'s Stage B re-run (before vs. after)
for the empirical counterpart of this argument.

**The separate FIFO fix** (`_BucketQueue`: `list`/`.pop()` → `deque`/
`.popleft()`, both O(1), see `docs/cpp_provenance.md`) was verified the
same way: static argument (only which end is dequeued changes; insert and
pop both stay O(1)), op counts before/after on F1/F2/F5 (same matching
size every time; other counters shift by <5%, consistent with finding a
different but equally-short augmenting path, not more work), identical
sizes on 2,452 graphs including all 52 known cases, and F5's edge-scan-
start iteration counts (69/97/139 at n_param 5000/10000/20000) unchanged.

---

## PROBLEM items (short list)

| #   | Item                                                                                     | Status                                                                                                                    | Suggested fix (not applied)                                                                                                                                                                                                                                                                                                            |
| --- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `tmp.insert(0, cur)` in Phase 2's blossom-absorption loop (§2c)                          | **Already fixed** (merged from Maya's change, now `tmp.append(cur)`)                                                      | None needed — already resolved. If ever reverted, replace with `append` + reverse the two downstream `for node in tmp:` loops, or use `collections.deque` with `appendleft` if front-order must be preserved for some future reason.                                                                                                   |
| 2   | Per-iteration O(n) rebuild of `T`/`label`/`bd`/`bDelta`/`base`/`dbase`/bucket array (§3) | Known, disclosed, **not fixed**                                                                                           | Mirror the C++ reference: make `T` (and the label/dual bookkeeping keyed off it) a value threaded between `phase1()` calls instead of rebuilt, trimming/growing it incrementally; would reduce the O(n·iterations) term to O(n) amortized over the whole run. Nontrivial refactor — deferred per this task's scope (no logic changes). |
| 3   | Ancestor-climb / blossom-absorption amortized bound (§2a)                                | Not a defect — but flagged because it's **unverified beyond re-deriving the standard argument from the code's structure** | If this bound matters for a real release, worth an explicit unit test or counter-based check (Stage B territory, deliberately not run here) that directly measures total climb-steps per iteration against n, rather than relying on this audit's structural argument alone.                                                           |

No other PROBLEM items found.

---

## Verdict

**Verified directly from the code in this audit:**

- Every loop whose body could plausibly run more than O(1) times per
  iteration either has an explicit O(m) or O(n) total-per-iteration bound
  derivable from a direct read (edge-scan counting, vertex-disjoint-path
  arguments), or an O(n)-amortized-total bound via the blossom-absorption
  charging argument (§2a), which is structurally sound (union happens
  inside the climbing loop, immediately) but is a restated standard
  argument, not a newly-constructed proof.
- Union-find is confirmed to be union-by-size + path compression, with
  every lookup going through `find`, and `make_rep` adding only O(1)
  marginal cost — i.e., O(α(n)) per operation, matching the paper's
  "O(m·α(n))" variant (not the unsophisticated raw-O(m) Gabow-Tarjan
  incremental-tree structure, which this does not implement, consistent
  with prior rounds).
- No recursion anywhere, direct or mutual; call-stack depth is O(1),
  independent of n and m.
- One genuine latent super-linear cost was found and is already fixed in
  the merged code (`tmp.insert(0, cur)` → `tmp.append(cur)`), and does not
  affect correctness.
- The per-iteration O(n) re-initialization of several structures is real
  and disclosed; it is exactly why the bound below is stated with an
  explicit `+n` term rather than the often-abbreviated O(m·α(n)). It is
  not amortized across iterations the way the C++ reference's structure
  is.

**Not verified from the code, and explicitly out of scope for this static
audit — depends on cited theory, not on anything measured or proven here:**

- The claim that the **number of iterations is O(√n)**. This audit
  establishes "O((n+m)·α(n)) **per** iteration" directly from the code; it
  says nothing about _how many_ iterations occur on any given input. That
  bound is the classical Hopcroft-Karp/Karzanov phase-count argument, cited
  (not reproved) by the Gabow (2017) paper itself (Sec. 2) and relied on
  here by reference, not rederived from this code. The prior round's
  investigation measured only **2** iterations on the paper's own "short
  and long chains" adversarial family, where the paper's Table 1 reports
  47 at a matching (n, m). To be precise about what that result does and
  does not show: 2 ≤ 47 ≤ O(√n), so it is **not** a violation of the O(√n)
  _upper_ bound — fewer iterations than predicted is never a
  counterexample to an upper bound. What it _does_ show is that this
  specific attempted reconstruction of the paper's worst-case family
  failed to **reproduce** the paper's worst-case behavior, so this project
  currently has **no empirical evidence that the O(√n) bound is tight** —
  i.e. no confirmed instance where the iteration count actually approaches
  √n — only (so far) instances consistent with it being a loose upper
  bound in practice. That remains an open, unresolved issue; this static
  audit neither confirms nor refutes it, since it is purely about
  per-iteration cost, not iteration count.

**Overall:** as written, one iteration (Phase 1 + Phase 2) costs
**O((n+m)·α(n))** — written with the explicit `+n` term (rather than the
often-abbreviated O(m·α(n))) because §3 identifies a real O(n) per-iteration
re-initialization cost that is not dominated by m on graphs where m < n —
and everything outside the iteration loop costs **O(n+m)**, both confirmed
directly from this static reading of the code. Combined with the
separately-cited (not code-verified) O(√n) iteration count, this would give
a total bound of **O(√n · (n+m) · α(n))** — the paper's own O(√n·m·α(n))
with the same `(n+m)` correction applied throughout. As is standard in the
literature, assuming no isolated vertices (m ≥ n/2) this equals
O(√n · m · α(n)), the form used in the paper. This combination has
not been demonstrated to hold on every input in this codebase: it is
consistent with everything checked here, and with this project's own
empirical measurements on random and several adversarial graph families,
but — per the point above — it remains **unconfirmed as a _tight_ bound**:
this project has not yet produced an input on which the iteration count is
observed to grow like √n, so while nothing found so far contradicts
O(√n·(n+m)·α(n)) as an upper bound, its tightness is still an open
question, not something this project has demonstrated.
