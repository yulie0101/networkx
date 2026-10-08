# C++ reference provenance and known deviations

`max_cardinality_matching_gabow` (`networkx/algorithms/matching.py`) is
ported from a LEDA-based C++ implementation of Gabow's algorithm [1]. This
document records exactly which C++ file that is, where it and its
companion files came from, and the full list of checked, deliberate or
tie-break-only deviations between this port and that file, found over the
course of porting and a subsequent dedicated investigation ("Task B").

The C++ files themselves are **never** copied into this repository or any
of its history; this document cites them by name and hash only.

## 1. Identity of the source file

The original C++ file provided for this port, `max_matching_c_2plus.cpp`
(20,871 bytes, SHA-256
`d800057b7d5b9b4a52066180ecda0929cab59373206b54b1b6f27075e7947280`), is
**content-identical** (601 non-blank lines, zero differences, modulo blank
lines and CRLF) to `GabowBeautified.h` from Kurt Mehlhorn's companion
page for his and Nobahari's "Gabow's O(sqrt(n) m) Maximum Cardinality
Matching Algorithm, Revisited" [2] implementation:

- Companion page, titled "Gabow's General Matching Algorithm:
  Implementation, Engineering, and Experiments":
  `https://people.mpi-inf.mpg.de/~mehlhorn/CompanionPageGenMatchingImplementation.html`.
  It lists `GabowBeautified.h` as "The Program (extracted from 1)", where
  1 is ADM24 [3].
- `GabowBeautified.h`:
  `https://people.mpi-inf.mpg.de/~mehlhorn/GabowBeautified.h` -- 19,539
  bytes, SHA-256
  `d770736c8378a566ca1106ec66b96af6710b71e80af0006e8eb630e89acf33e5`.
  **This is the file cited throughout this document and in code comments
  as "the C++ reference".**
- `GabowRevised.h`:
  `https://people.mpi-inf.mpg.de/~mehlhorn/GabowRevised.h` -- 21,327
  bytes, SHA-256
  `ab512f1759a49bd9b4c04b8e3be6716b19bf9688c6a7fa2ce8793af70f73c764`. A
  **different, later file** (a separate `lcp`/`lcp_odd` reformulation of
  Part I, class `G_card_matching_GR` vs. `GabowBeautified.h`'s
  `G_card_matching`) -- cited only for contrast, never as "the reference"
  on its own.

Both, plus five other companion-page files not load-bearing for this
project (`_mc_matching.cpp`, `class_new_mc_matching.cpp`, `KurtTest.cpp`,
`mc_timing_long_chains.cpp`, `mc_timing_short_chains.cpp`), were downloaded
from the companion page above, by direct URL per file
(`https://people.mpi-inf.mpg.de/~mehlhorn/<filename>`), on 2026-10-06.

Two other papers describe this same implementation and are cited below by
their arXiv IDs:

- **[2] Mehlhorn & Nobahari**, "Gabow's O(sqrt(n) m) Maximum Cardinality
  Matching Algorithm, Revisited", arXiv:2603.22909 -- the algorithm
  write-up this port follows for terminology, counters, and
  instrumentation.
- **[3] Ansaripour, Danaei & Mehlhorn ("ADM24")**, "Gabow's Cardinality
  Matching Algorithm in General Graphs: Implementation and Experiments",
  arXiv:2409.14849 -- describes `GabowBeautified.h` itself in detail,
  including a pseudocode listing essentially identical to it.

## 2. Known deviations from `GabowBeautified.h`

Everything below was found and checked while porting, or during Task B (a
dedicated investigation into one specific discrepancy, described in
§3). None is known to change the _size_ of the returned matching.

### 2.1 Fixed: the Delta-bucket queue was LIFO, not FIFO

`GabowBeautified.h`'s `simple_queue` inserts via LEDA `list::append`
(back) and removes via LEDA `list::pop` (removes the **first** element --
a genuine FIFO queue, consistent with the class's own name). This port's
`_BucketQueue` originally removed via Python `list.pop()` with no index
-- the same end as `.append()`, making it a LIFO stack. This was the
actual translation error behind the stale-bucket-entry bug (see §3 and
`docs/complexity_audit.md` §8); fixed by switching `_BucketQueue.buckets`
to `collections.deque` and using `.popleft()`.

### 2.2 Kept: EVEN-branch tightness recheck

A popped bucket entry is rechecked for tightness (`d(x) + d(y) == 0`)
before being treated as a blossom step or augmenting path.
`GabowBeautified.h` has no such recheck, and Task B found this check is
not actually what causes the stale-bucket-entry bug (see §3) -- but the
recheck is correct, cheap (no change to the per-popped-entry complexity,
see `docs/complexity_audit.md` §8), and is kept as a safeguard. It mirrors
the guard present in the later `GabowRevised.h`
(`lcp[x] + lcp[y] == 2*Delta - 2`), whose bridge-step is defined as "while
there exists a tight even-even edge" -- a live recheck by construction.

### 2.3 Tie-breaks only: `T` and Phase 2's root order

- `T` (the search-structure list): this port appends where
  `GabowBeautified.h` prepends (`T.push`).
- Phase 2's root-selection loop: this port iterates
  `contracted_into`'s keys in first-occurrence-while-building order;
  `GabowBeautified.h` scans `T` directly, filtered to self-reps
  (`forall(vh, T) { if (vh != rep[vh]) continue; ...}`). This one was
  found only while building the faithful-order cross-check below, not
  during the original port or during the main Task B investigation.

**Not a deviation: `tmp`** (Phase 2's blossom-absorption list). This port
appends where `GabowBeautified.h` prepends (`tmp.push_front`), but the
reference then explores `tmp` by recursion (`forall(zh, tmp)
find_apHG(zh)`, commented "the new even node closest to bh comes first")
while this port pushes `tmp` onto an explicit LIFO stack. Appending plus
the stack's reversal gives the same order as prepending plus recursion:
both explore u_1, the vertex closest to the blossom base, first, as
Gabow's Fig. 4 line 6 prescribes. Run traces confirm it (blossom steps
absorbing at least 2 vertices on 3,000 random graphs: this port u_1 first
in 114/114, the compiled reference u_1 first in 55/55).

A **faithful-order cross-check** -- a scratch copy of this port with the
two orders above changed to match `GabowBeautified.h`, run against the
compiled, unmodified (except making its members public, for test-harness
access) reference C++ on identical inputs (the 52 originally
known-failing cases, a 6,000-graph battery, and small F3/F4/F5 instances,
with two initial matchings each) -- found **all 6,320 cases identical**
edge-for-edge, in iteration count, and in augmentations per iteration.
An earlier version of this cross-check also changed `tmp` to prepend,
which (because of the stack) reversed its order relative to the
reference, and found 6,314 of 6,320 identical; those 6 differences were
caused by that reversal and disappear without it.

### 2.4 Checked, not the cause: `bd`/`bDelta` initialization

This port resets `bd`/`bDelta` to 1/0 for every vertex at the start of
_every_ iteration of the outer loop, following ADM24's stated convention
("we initialize d(v) to one for all v"). `GabowBeautified.h`'s comment
states its own `bd`/`bDelta` ("...and there is no need to reinitialize
bd, bDelta, parent, source_bridge, and target_bridge") persist across
iterations as class members, and -- confirmed by direct inspection of the
compiled shim's `node_array<int>` default -- a vertex untouched since the
start of the whole run reads 0, not 1 (real LEDA's default was not
independently verified).

The faithful-order cross-check (§2.3) is identical on all 6,320 cases
without changing this, so the per-iteration reset makes no observable
difference there. (Earlier, with the older cross-check's reversed `tmp`,
forcing the copy to persist `bd`/`bDelta`/`parent`/`source_bridge`/
`target_bridge` across iterations, or to default untouched vertices to 0,
did not change its 6 differing cases either; those were later traced to
the `tmp` reversal, see §2.3.) This difference is recorded here as
disclosed and checked, not as a cause of anything.

### 2.5 Kept: the `gained == 0` guard now raises

If Phase 1 finds an augmenting path (`found_sap`) but Phase 2 finds no
augmenting path in H, this is a violation of Corollary 3.3 and should be
unreachable. This port now raises `NetworkXAlgorithmError` when it
happens, rather than silently breaking out of the main loop (which would
return a matching that is not maximum, with no indication anything went
wrong). `GabowBeautified.h`'s `solve()` has no equivalent check at all --
its loop (`if (phase_1()) phase_2(); else break;`) would retry
indefinitely if this ever happened with its own (FIFO) queue.

### 2.6 Structural, not order: iterative instead of recursive helpers

Every helper `GabowBeautified.h` implements via native C++ recursion
(Phase 2's search, and the two path-reconstruction routines) is
implemented here with an explicit stack instead, since Python's call
stack is far shallower than the O(n)-deep recursions a large sparse graph
can trigger. Converting recursion to an explicit stack requires reversing
push order for children to preserve the same depth-first exploration
order (see the comment at the Phase 2 blossom-absorption site in
`matching.py`, which documents this precisely using Mehlhorn & Nobahari's
own u_1..u_k notation).

### 2.7 Deliberate: no heuristic fallback

`GabowBeautified.h`'s own timing driver (`solve()`) enables a single-
augmenting-path-at-a-time fallback by default once few augmentations
remain. That fallback is not part of the algorithm in [1] and, ported
faithfully, would abandon the `O(sqrt(n) * m * alpha(n))` bound (up to one
iteration per remaining augmenting path once triggered, rather than one
per distinct shortest-path length) -- so it is intentionally not offered
in this port, not even as an opt-in.

## 3. Task B: the stale-bucket-entry bug, fully attributed

An independent review found a correctness bug (52 failures out of 120,000
random-graph trials) and the original fix added the tightness recheck in
§2.2, attributing the bug to a "translation miss" of that recheck. A
later, dedicated investigation ("Task B") checked this attribution
directly against the compiled reference and found it was not quite right:

- `GabowBeautified.h`'s EVEN branch has no recheck either -- the
  structural omission described in §2.2 is not unique to this port.
- Compiling `GabowBeautified.h` unmodified (members made public only, for
  harness access) and running it with this port's own node/edge order and
  initial matchings reproduced **zero** of the 52 originally-failing
  cases, and zero failures across a further 226,143-case sweep (random
  graphs n=6-60, several densities, 3 node orders, 3 initial-matching
  strategies, plus adversarial families).
- The actual, confirmed cause is §2.1 (LIFO vs. FIFO). A FIFO-only patch
  of the pre-fix Python (recheck still absent) also passed all 232,195
  combined test cases. Forcing _only_ the compiled reference's bucket to
  LIFO (isolated from every other list in the class) reproduced the
  original bug exactly, including on the minimal 10-node example that
  originally surfaced it.
- Whether the recheck is also necessary once the queue is FIFO (i.e.
  whether it is a latent bug in `GabowBeautified.h` too, simply never
  triggered) remains formally open: neither Mehlhorn & Nobahari [2] nor
  ADM24 [3] state an invariant that rules out a stale entry for this
  specific `bd`/`bDelta`-based formulation. The 232,195-case sweep above
  is strong empirical evidence, not a proof.

The tightness re-check (§2.2) is correct; it was necessary with the
port's former LIFO queue, and is kept as a safeguard. The FIFO fix (§2.1)
is a separate, independently-applied and independently-verified change
(see `docs/complexity_audit.md` §8 for its own complexity verification).

## References

1. Harold N. Gabow, "The Weighted Matching Approach to Maximum
   Cardinality Matching", Fundamenta Informaticae 154 (2017), 109-130.
2. Kurt Mehlhorn, Romina Nobahari, "Gabow's O(sqrt(n) m) Maximum
   Cardinality Matching Algorithm, Revisited", arXiv:2603.22909.
3. Matin Ansaripour, Alireza Danaei, Kurt Mehlhorn, "Gabow's Cardinality
   Matching Algorithm in General Graphs: Implementation and Experiments",
   arXiv:2409.14849.
