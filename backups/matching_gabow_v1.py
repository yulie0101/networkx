"""Frozen backup of max_cardinality_matching_gabow, as it stood at the
"gabow-v1-backup" git tag, BEFORE adding union-by-size to _UnionFind and
before investigating the short-and-long-chains iteration-count discrepancy.

This file is a snapshot, not a living module: do not edit it. Its purpose
is to let the current implementation (networkx.algorithms.matching) be
compared side by side against this known-good prior version -- e.g. to
confirm a later change (such as adding union-by-size) doesn't alter which
matching is returned, or to compare operation counts before/after.

The function is renamed to `max_cardinality_matching_gabow_v1` (otherwise
unchanged, including its own docstring, which still refers to the
behavior by its original name in a few places) so both versions can be
imported and called in the same script without clashing:

    from networkx.algorithms.matching import max_cardinality_matching_gabow
    from backups.matching_gabow_v1 import max_cardinality_matching_gabow_v1

See also: backups/test_matching_v1_additions.py and
backups/bench_matching_v1.py / backups/verify_complexity_v1.py for the
tests and benchmark scripts that accompanied this version.
"""

from networkx.utils import not_implemented_for
from networkx.algorithms.matching import matching_dict_to_set

import networkx as nx

@not_implemented_for("multigraph")
@not_implemented_for("directed")
@nx._dispatchable
def max_cardinality_matching_gabow_v1(
    G, use_heuristic_fallback=False, _counters=None, _skip_greedy_init=False
):
    r"""Compute a maximum cardinality matching using Gabow's algorithm.

    A matching is a subset of edges in which no node occurs more than once.
    This function returns a matching of maximum cardinality (the greatest
    possible number of edges), using the algorithm of Gabow [1]_, which
    recasts Edmonds' blossom method in terms of maximum *weight* matching:
    every iteration weighs matched edges 2 and unmatched edges 0, so that a
    maximum weight augmenting path is automatically a *shortest* augmenting
    path (see Notes). Repeatedly finding a maximal collection of
    vertex-disjoint shortest augmenting paths and augmenting along all of
    them (the classical Hopcroft-Karp/Karzanov strategy) yields a maximum
    cardinality matching in :math:`O(\sqrt{n} \cdot m)` time.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph. Self-loops are ignored (a self-loop can never be
        part of a matching); multigraphs and directed graphs are rejected.

    use_heuristic_fallback : bool, optional (default=False)
        If True, once a large fraction of the estimated remaining
        augmentations have already been spent on full phases, switch to
        completing the matching by finding one augmenting path at a time
        instead of a full maximal disjoint set per phase. This mirrors a
        practical optimization in Gabow's reference implementation for the
        cheap "tail" of the computation, where a full phase's bookkeeping
        overhead is not repaid by the few augmenting paths it finds. It
        never changes the *result* (see Notes) and defaults to off so the
        algorithm run is a direct, inspectable translation of Fig. 1 of the
        paper.

    _skip_greedy_init : bool, optional (default=False)
        Internal instrumentation hook, not part of the public API. If True,
        skip the greedy warm-start below and begin from M = (empty set),
        exactly as Fig. 1 specifies -- used by the complexity-verification
        benchmark to measure the paper's bare algorithm on adversarial
        inputs designed assuming an empty start; the greedy warm-start
        (itself not in the paper) can trivially pre-solve some such inputs,
        which would make a phase-count experiment about Fig. 1 vacuous.
        Never used by default and never changes behavior unless passed.

    _counters : dict, optional (default=None)
        Internal instrumentation hook, not part of the public API (may
        change or be removed without notice). If a dict is given, it is
        populated in place with operation counts used to empirically check
        this function's time complexity -- see the "Instrumentation" note
        below. Passing it does not change the algorithm's logic, control
        flow, or returned matching in any way; every counter update is an
        isolated ``+= 1`` beside an existing statement, guarded by
        ``if _counters is not None``, so the default (``None``) path pays
        only that one extra check per site.

    Returns
    -------
    matching : set
        A maximum cardinality matching of the graph, as a set of edges.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> len(nx.max_cardinality_matching_gabow_v1(G))
    2

    Notes
    -----
    **Algorithm.** Each outer iteration ("phase", Fig. 1 of [1]_) runs two
    steps:

    * *Phase 1* (Sec. 3) is a single search of Edmonds' weighted matching
      algorithm specialized to weights of 2 (matched edges) and 0 (unmatched
      edges) with constant initial duals. Because every free vertex keeps
      the same dual value throughout a search, the dual adjustment amount
      can be tracked as one global counter :math:`\Delta` instead of
      updating every vertex's dual separately (Sec. 5, "Dual adjustment
      steps"); edges are kept in a bucket queue indexed by the value of
      :math:`\Delta` at which they become tight. Lemma 3.1 and the fact
      that an augmenting path's length is ``-w(P) + 1`` (eq. 3) together
      show :math:`\Delta` never exceeds :math:`n/2`, so the bucket array has
      O(n) slots and every edge is inserted/extracted O(1) times, giving
      O(m + n) work. Phase 1 finds a maximum *weight* augmenting path, which
      by the above is automatically a shortest augmenting path (an "sap"),
      and it also contracts every maximal *positive* blossom (one formed
      before the last dual adjustment) to build the minor graph H, whose
      augmenting paths correspond 1-to-1 to the shortest augmenting paths of
      G (Corollary 3.3).
    * *Phase 2* (Sec. 4, Fig. 4, restating Gabow and Tarjan's ``find_ap_set``
      procedure) runs a path-preserving depth-first search on H to find a
      *maximal* set of vertex-disjoint augmenting paths in one O(m) pass:
      blossom steps are delayed until a genuine ancestor relationship in the
      grow-forest is confirmed, which is tested in O(1) using an "even
      became outer" timestamp rather than an explicit ancestor walk
      (Appendix A here proves the O(1) test equivalent to the descendant
      test). Every edge is scanned at most twice (once from each endpoint),
      so this phase is also O(m). Each found H-path is lifted back to a
      real augmenting path of G using the blossom "bridge" pointers
      recorded during Phase 1's blossom-shrink steps, and the matching is
      updated along it.

    The number of phases until Phase 1 first fails to find an augmenting
    path is :math:`O(\sqrt{n})`; this is the classical Hopcroft-Karp/Karzanov
    phase-count bound (cited, not reproved, in [1]_, Sec. 2) and falls out
    automatically here because each phase both finds a *shortest* augmenting
    path and augments along a *maximal* disjoint set of such paths -- no
    separate bookkeeping is needed to obtain it. Combined with the O(m) cost
    per phase this gives the stated O(:math:`\sqrt{n} \cdot m`) total time
    (Theorem 5.1 of [1]_).

    **Implementation notes / deviations from the reference.** Blossom
    partitions here use a small union-find with path compression only (no
    union-by-rank, since the base vertex chosen when two blossoms merge is
    fixed by the algorithm rather than arbitrary), giving amortized
    O(:math:`\log n`) per operation. This is a practical stand-in for the
    specialized O(1) amortized incremental-tree union-find structure of
    Gabow and Tarjan used in the reference implementation and does not
    change the algorithm's overall complexity class. Every helper that the
    reference implements recursively (the Phase 2 search and the two
    path-reconstruction routines) is implemented here with an explicit
    stack instead, since Python's call stack is far shallower than the
    O(n)-deep recursions a large sparse graph (e.g. a long path or cycle)
    can trigger.

    ``use_heuristic_fallback=True`` does not reproduce the reference
    implementation's fallback exactly: rather than switching to a
    completely separate classical (non-phase, non-shortest-path) blossom
    search -- which would duplicate a second, independently-fallible
    implementation of blossom shrinking for a purely constant-factor gain
    -- it reuses the same, already-verified Phase 1/Phase 2 machinery but
    stops each Phase 2 call after its *first* augmenting path instead of
    continuing to find a maximal disjoint set. This still avoids paying for
    an exhaustive Phase 2 sweep once few augmenting opportunities remain,
    without introducing a second augmenting-path algorithm; it is weaker
    than the reference's fallback (which also skips Phase 1's dual-adjustment
    bookkeeping per augmentation) but carries no additional correctness risk.
    Both settings always produce a matching of maximum cardinality.

    **Instrumentation (``_counters``).** Following Mehlhorn & Nobahari's
    terminology [2]_ -- "iteration" for the outer loop (Fig. 1's ``loop``,
    O(sqrt(n)) of them) and "phase Delta" for the Delta-indexed steps inside
    Phase 1 of one iteration -- ``_counters`` is populated with: ``iterations``
    and ``delta_phases`` (total Delta-phases summed over all iterations);
    ``edge_scans`` and ``search_steps`` (Phase 1's ``scan_edge`` calls and
    Delta-bucket dequeues; Phase 2's per-edge DFS steps); ``blossom_
    contraction_events`` and ``blossom_shrink_vertex_steps`` (how many
    blossoms formed, and the per-vertex absorption work each one cost);
    ``blossom_expansions`` (always 0 -- the simplified search of Fig. 2,
    under assumption (A), never expands a blossom mid-search, since no
    phase carries blossoms over from a previous one); ``union_find_find_
    calls``/``union_find_find_hops``/``union_find_union_calls`` split by
    structure (``uf_base_*`` for Sec. 3.1's S-bar partition, ``uf_dbase_*``
    for the maximal-positive-blossom partition of Sec. 3.4, which Phase 2
    continues to use -- see Notes); ``augmentations`` (Fig. 1's "augment M
    by Q"); and ``per_iteration``, a list with one entry per iteration
    holding that same set of keys scoped to just that iteration's work.

    References
    ----------
    .. [1] Harold N. Gabow, "The Weighted Matching Approach to Maximum
       Cardinality Matching", Fundamenta Informaticae 154 (2017), 109-130.
       https://doi.org/10.3233/FI-2017-1555
    .. [2] Kurt Mehlhorn, Romina Nobahari, "Gabow's O(sqrt(n) m) Maximum
       Cardinality Matching Algorithm, Revisited", arXiv:2603.22909.
    """
    n = G.number_of_nodes()
    if n == 0:
        return set()

    # Adjacency with self-loops removed once, up front (O(m)); self-loops
    # can never appear in a matching, so they are silently dropped rather
    # than rejected.
    adj = {v: [u for u in G[v] if u != v] for v in G}

    # `mate` follows the convention used by max_weight_matching: an
    # unmatched vertex is simply absent as a key (never mapped to None).
    mate = {}

    # A greedy matching gives a cheap 2-approximate upper bound on the true
    # maximum matching size (any maximal matching is at least half the
    # optimum). This bound is only used by the optional heuristic fallback
    # below to estimate how many augmentations are likely still needed.
    size_of_M = 0
    if not _skip_greedy_init:
        for v in G:
            if v in mate:
                continue
            for u in adj[v]:
                if u not in mate:
                    mate[v] = u
                    mate[u] = v
                    size_of_M += 1
                    break
    max_size_of_M = min(n // 2, 2 * size_of_M)
    number_of_iterations = 0

    if _counters is not None:
        _counters.update(
            {
                "iterations": 0,
                "delta_phases": 0,
                "edge_scans": 0,
                "search_steps": 0,
                "blossom_contraction_events": 0,
                "blossom_shrink_vertex_steps": 0,
                "blossom_expansions": 0,  # always 0 -- see docstring
                "uf_base_find_calls": 0,
                "uf_base_find_hops": 0,
                "uf_base_union_calls": 0,
                "uf_dbase_find_calls": 0,
                "uf_dbase_find_hops": 0,
                "uf_dbase_union_calls": 0,
                "augmentations": 0,
                "per_iteration": [],
            }
        )

    def _snapshot():
        return {k: v for k, v in _counters.items() if k != "per_iteration"}

    def _diff(before, after):
        return {k: after[k] - before[k] for k in before}

    class _UnionFind:
        """Union-find with path compression, supporting a forced
        representative after a union (``make_rep``) -- needed because a
        blossom's base vertex must remain the partition's representative,
        which plain union-by-rank union-find does not guarantee. A fresh
        instance is created for every Phase 1 call, scoped implicitly to
        whichever vertices are touched; this costs O(n) to allocate, which
        summed over the O(sqrt(n)) phases contributes O(n**1.5) additional
        overhead -- within the O(sqrt(n) * m) bound whenever m = Omega(n),
        i.e. for any graph whose matching structure actually requires
        multiple phases.
        """

        __slots__ = ("_name", "parent")

        def __init__(self, elements, name="base"):
            self.parent = {v: v for v in elements}
            self._name = name  # "base" or "dbase", for _counters keys only

        def find(self, x):
            if _counters is not None:
                _counters["uf_" + self._name + "_find_calls"] += 1
            parent = self.parent
            root = x
            while parent[root] != root:
                root = parent[root]
                if _counters is not None:
                    _counters["uf_" + self._name + "_find_hops"] += 1
            while parent[x] != root:
                parent[x], x = root, parent[x]
                if _counters is not None:
                    _counters["uf_" + self._name + "_find_hops"] += 1
            return root

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.parent[rx] = ry
                if _counters is not None:
                    _counters["uf_" + self._name + "_union_calls"] += 1

        def make_rep(self, x):
            r = self.find(x)
            if r != x:
                self.parent[r] = x
                self.parent[x] = x
                if _counters is not None:
                    _counters["uf_" + self._name + "_union_calls"] += 1

    class _BucketQueue:
        """Bucket priority queue indexed by dual-adjustment level Delta
        (Sec. 5, "Dual adjustment steps"). Since Delta never exceeds n/2,
        a fixed-size array of buckets gives O(1) insert/pop.
        """

        __slots__ = ("buckets", "size")

        def __init__(self, size):
            self.size = size
            self.buckets = [[] for _ in range(size)]

        def insert(self, edge, d):
            if d < self.size:
                self.buckets[d].append(edge)

        def pop(self, d):
            if d >= self.size or not self.buckets[d]:
                return None
            return self.buckets[d].pop()

    def phase1():
        """One search of Edmonds' algorithm (paper Sec. 3, Fig. 2)
        specialized to weight-2/weight-0 cardinality matching. Returns a
        dict describing the minor graph H if a maximum weight (= shortest)
        augmenting path is found, else None.
        """
        # `label[v]` is "EVEN"/"ODD" for v's *current* blossom base once
        # known, "UNLABELED" otherwise; free vertices start EVEN, every
        # other vertex starts UNLABELED (matched vertices are only ever
        # discovered by a grow step, never reset back to UNLABELED).
        label = {v: ("EVEN" if v not in mate else "UNLABELED") for v in G}
        parent = {}
        # bd[v]/bDelta[v] let the current dual value of v be recovered
        # lazily as bd[v] +/- (Delta - bDelta[v]) (sign per label), instead
        # of updating every vertex's dual on every dual adjustment step.
        bd = {v: 1 for v in G}
        bDelta = {v: 0 for v in G}
        # Bridges recorded while shrinking a blossom: for v inside a
        # blossom with base b, (source_bridge[v], target_bridge[v]) is the
        # non-matching edge that closes the blossom on v's side of it.
        source_bridge = {}
        target_bridge = {}
        # `base` is Edmonds' S-bar blossom partition, live during this
        # search; `dbase` accumulates the *maximal positive* blossoms
        # (Sec. 3.4) and is committed to only once per Delta level.
        base = _UnionFind(G, name="base")
        dbase = _UnionFind(G, name="dbase")
        path1 = {}
        path2 = {}
        strue = 0
        T = [v for v in G if v not in mate]
        queue = _BucketQueue(n // 2 + 1)
        Delta = 0

        def d(v):
            lbl = label[base.find(v)]
            if lbl == "UNLABELED":
                return 1
            if lbl == "EVEN":
                return bd[v] - (Delta - bDelta[v])
            return bd[v] + (Delta - bDelta[v])

        def scan_edge(u, z):
            # Scan non-matching edge zu from z, which just became EVEN (or
            # was just absorbed into a blossom with base EVEN).
            if _counters is not None:
                _counters["edge_scans"] += 1
            if mate.get(u) == z or label[base.find(u)] == "ODD":
                return
            p = d(z) + d(u)
            if label[u] == "UNLABELED":
                queue.insert((z, u), Delta + p)
            else:
                queue.insert((z, u), Delta + p // 2)

        def shrink_path(b, x, y, dunions):
            v = base.find(x)
            while v != b:
                if _counters is not None:
                    _counters["blossom_shrink_vertex_steps"] += 1
                base.union(v, b)
                dunions.append(v)
                dunions.append(b)
                v = mate[v]
                base.union(v, b)
                dunions.append(v)
                dunions.append(b)
                base.make_rep(b)
                source_bridge[v] = x
                target_bridge[v] = y
                bd[v] = bd[v] + (Delta - bDelta[v])
                bDelta[v] = Delta
                for u in adj[v]:
                    scan_edge(u, v)
                v = base.find(parent[v])
            dunions.append(b)
            dunions.append(b)

        for v in T:
            for u in adj[v]:
                scan_edge(u, v)

        found_sap = False
        dunions = []
        while 2 * Delta <= n:
            if _counters is not None:
                _counters["delta_phases"] += 1
            while True:
                e = queue.pop(Delta)
                if e is None:
                    break
                if _counters is not None:
                    _counters["search_steps"] += 1
                x, y = e
                if label[base.find(x)] != "EVEN":
                    x, y = y, x
                bx, by = base.find(x), base.find(y)
                if y == mate.get(x) or bx == by or label[by] == "ODD":
                    continue
                if label[by] == "UNLABELED":
                    # Grow step.
                    z = mate[y]
                    bd[y] = bd[z] = 1
                    bDelta[y] = bDelta[z] = Delta
                    parent[z] = y
                    parent[y] = x
                    label[y] = "ODD"
                    label[z] = "EVEN"
                    T.append(y)
                    T.append(z)
                    for u in adj[z]:
                        scan_edge(u, z)
                else:  # label[by] == "EVEN": blossom step or augmenting path
                    strue += 1
                    hx, hy = bx, by
                    path1[hx] = strue
                    path2[hy] = strue
                    while (path1.get(hy) != strue and path2.get(hx) != strue) and (
                        mate.get(hx) is not None or mate.get(hy) is not None
                    ):
                        if mate.get(hx) is not None:
                            hx = base.find(parent[mate[hx]])
                            path1[hx] = strue
                        if mate.get(hy) is not None:
                            hy = base.find(parent[mate[hy]])
                            path2[hy] = strue
                    if path1.get(hy) == strue or path2.get(hx) == strue:
                        if _counters is not None:
                            _counters["blossom_contraction_events"] += 1
                        b = hy if path1.get(hy) == strue else hx
                        shrink_path(b, x, y, dunions)
                        shrink_path(b, y, x, dunions)
                    else:
                        found_sap = True
                # Note: processing continues even after found_sap becomes
                # True, draining every tight edge at this Delta level --
                # H is defined by the *maximal* positive blossoms as of
                # immediately before the last dual adjustment (Sec. 5,
                # "Constructing graph H"), so every blossom step at the
                # final Delta must be completed before H is built below.
            if found_sap:
                break
            i = 0
            while i < len(dunions):
                u, v = dunions[i], dunions[i + 1]
                if u == v:
                    dbase.make_rep(u)
                else:
                    dbase.union(u, v)
                i += 2
            dunions.clear()
            Delta += 1

        if not found_sap:
            return None

        # Build H: contract every maximal positive blossom, keep the tight
        # edges joining distinct blossoms (Sec. 3.3 / Sec. 5).
        rep = {v: dbase.find(v) for v in T}
        contracted_into = {}
        mateHG = {}
        is_edge_of_H = set()
        for v in T:
            contracted_into.setdefault(rep[v], []).append(v)
            mateHG.setdefault(rep[v], None)
        for u in T:
            uh = rep[u]
            for v in adj[u]:
                vh = rep.get(v)
                if vh is None or uh == vh:
                    continue
                w_e = 2 if mate.get(u) == v else 0
                if d(u) + d(v) == w_e:
                    is_edge_of_H.add((u, v))
                    if w_e == 2:
                        mateHG[uh] = vh
                        mateHG[vh] = uh
        return {
            "rep": rep,
            "contracted_into": contracted_into,
            "mateHG": mateHG,
            "is_edge_of_H": is_edge_of_H,
            "dbase": dbase,
            "label": label,
            "parent": parent,
            "source_bridge": source_bridge,
            "target_bridge": target_bridge,
        }

    def phase2(H, stop_after_first):
        """Path-preserving depth-first search on H (paper Sec. 4, Fig. 4,
        Gabow-Tarjan's ``find_ap_set``/``find_ap``) to find a maximal set
        of vertex-disjoint augmenting paths, lift each one back to G and
        augment along it. Returns the number of augmentations performed.

        H's own blossom partition continues to use Phase 1's `dbase`
        object: an H-node is exactly a `dbase` representative from Phase 1,
        and a Phase 2 blossom step only ever merges such representatives
        further, so no separate union-find is needed (mirrors the
        reference implementation reusing the same partition).
        """
        rep = H["rep"]
        contracted_into = H["contracted_into"]
        mateHG = H["mateHG"]
        is_edge_of_H = H["is_edge_of_H"]
        dbase = H["dbase"]
        label = H["label"]
        parent = H["parent"]
        source_bridge = H["source_bridge"]
        target_bridge = H["target_bridge"]

        labelHG = {}
        parentHG = {}
        even_timeHG = {}
        bridgeHG = {}
        tG = 0

        def edge_iter_for(vh):
            for v in contracted_into[vh]:
                for u in adj[v]:
                    if (v, u) in is_edge_of_H:
                        yield (v, u)

        def trace_HG(start_vh, start_uh):
            """Recover the H-edges of the alternating path from start_vh to
            start_uh (order does not matter -- augmenting only needs the
            *set* of edges to flip, see augment()). Iterative version of
            the paper's ``find_path_in_HG``.
            """
            edges = []
            stack = [(start_vh, start_uh)]
            while stack:
                vh, uh = stack.pop()
                if vh == uh:
                    continue
                if labelHG.get(vh) == "EVEN":
                    mvh = mateHG[vh]
                    pv, pu = parentHG[mvh]
                    edges.append((pv, pu))
                    other = pu if rep[pv] == mvh else pv
                    stack.append((rep[other], uh))
                else:
                    bridge_v, bridge_u = bridgeHG[vh]
                    edges.append((bridge_v, bridge_u))
                    stack.append((rep[bridge_u], rep[mateHG[vh]]))
                    stack.append((rep[bridge_v], uh))
            return edges

        def trace_G(start_v, start_u, mate_pairs):
            """Recover the pairs to (re)match along the even-length
            alternating path from start_v to start_u inside a single
            Phase 1 blossom, via its recorded bridges. Iterative version
            of the paper's ``find_path_in_G``.
            """
            stack = [(start_v, start_u)]
            while stack:
                v, u = stack.pop()
                if v == u:
                    continue
                if label[v] == "EVEN":
                    mv = mate[v]
                    pv = parent[mv]
                    mate_pairs.append((mv, pv))
                    stack.append((pv, u))
                else:
                    sb = source_bridge[v]
                    tb = target_bridge[v]
                    mate_pairs.append((sb, tb))
                    stack.append((sb, mate[v]))
                    stack.append((tb, u))

        def augment(h_edges):
            mate_pairs = []
            for u, v in h_edges:
                mate_pairs.append((u, v))
                trace_G(u, rep[u], mate_pairs)
                trace_G(v, rep[v], mate_pairs)
            for a, b in mate_pairs:
                mate[a] = b
                mate[b] = a

        augmentations = 0
        for root in list(contracted_into):
            if stop_after_first and augmentations > 0:
                break
            if mateHG.get(root) is not None or labelHG.get(root) is not None:
                continue
            labelHG[root] = "EVEN"
            even_timeHG[root] = tG
            tG += 1

            stack = [[root, edge_iter_for(root)]]
            found_h_edges = None
            while stack:
                vh, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    continue
                if _counters is not None:
                    # Phase 2's DFS has no separate generate/consume split
                    # like Phase 1 -- examining an edge here is both the
                    # "search step" and the "edge scan" in one.
                    _counters["search_steps"] += 1
                    _counters["edge_scans"] += 1
                v, u = nxt
                uh = rep[u]
                if mateHG.get(vh) == uh:
                    continue
                if labelHG.get(uh) is None:
                    parentHG[uh] = (v, u)
                    mateHG_uh = mateHG.get(uh)
                    if mateHG_uh is None:
                        # y is free: xy plus the tree paths to the root
                        # form an augmenting path (Fig. 4, line 2).
                        labelHG[uh] = "ODD"
                        found_h_edges = [(v, u)] + trace_HG(rep[v], root)
                        break
                    labelHG[uh] = "ODD"
                    labelHG[mateHG_uh] = "EVEN"
                    even_timeHG[mateHG_uh] = tG
                    tG += 1
                    stack.append([mateHG_uh, edge_iter_for(mateHG_uh)])
                else:
                    bh = dbase.find(vh)
                    zh = dbase.find(uh)
                    if bh != zh and even_timeHG.get(bh, -1) < even_timeHG.get(zh, -1):
                        # Blossom step (Fig. 4, line 3 "equivalent test":
                        # bh's blossom became outer strictly before zh's).
                        if _counters is not None:
                            _counters["blossom_contraction_events"] += 1
                        tmp = []
                        endpoints_of_M = []
                        cur = zh
                        while cur != bh:
                            if _counters is not None:
                                _counters["blossom_shrink_vertex_steps"] += 1
                            endpoints_of_M.append(cur)
                            cur = mateHG[cur]
                            endpoints_of_M.append(cur)
                            tmp.insert(0, cur)
                            pv, pu = parentHG[cur]
                            other = pu if rep[pv] == cur else pv
                            cur = dbase.find(rep[other])
                        for node in endpoints_of_M:
                            dbase.union(node, bh)
                        dbase.make_rep(bh)
                        for node in tmp:
                            bridgeHG[node] = (v, u)
                        for node in tmp:
                            stack.append([node, edge_iter_for(node)])
            if found_h_edges is not None:
                augment(found_h_edges)
                augmentations += 1
                if _counters is not None:
                    _counters["augmentations"] += 1

        return augmentations

    while True:
        number_of_iterations += 1
        if _counters is not None:
            _counters["iterations"] += 1
            iter_before = _snapshot()
        H = phase1()
        if H is None:
            if _counters is not None:
                _counters["per_iteration"].append(_diff(iter_before, _snapshot()))
            break
        stop_after_first = use_heuristic_fallback and (
            number_of_iterations > 0.5 * (max_size_of_M - size_of_M)
        )
        gained = phase2(H, stop_after_first)
        size_of_M += gained
        if _counters is not None:
            _counters["per_iteration"].append(_diff(iter_before, _snapshot()))
        if gained == 0:
            # Phase 1 found an sap, so Phase 2 must find at least one
            # augmenting path in H (Corollary 3.3); this is a defensive
            # guard against an infinite loop, not expected to trigger.
            break

    return matching_dict_to_set(mate)
