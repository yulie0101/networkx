"""Instrumented copy of ``max_cardinality_matching_gabow`` (final code).

Used to produce the worked examples in the report (Chapters 4-6). The
algorithm is the final version on this branch -- FIFO Delta-bucket queue
(``deque.popleft``), tightness re-check of popped EVEN-EVEN entries, union by
size, ``tmp`` explored from u_1, and an error if an iteration gains nothing.
The only additions are logging statements (and, when logging is on, a
parallel record of when and from where each bucket entry was inserted);
control flow and the returned matching are unchanged, which ``--verify``
checks against ``networkx.max_cardinality_matching_gabow``.

Usage::

    python tools/gabow_instr.py              # Examples A and B -> tools/output/
    python tools/gabow_instr.py --verify 1000

Each example writes a text log and a JSON file with the same events.
Spec: gabow_instr_spec.md (report folder).
"""

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set

OUTDIR = Path(__file__).parent / "output"


class Log:
    """Collects every event as a text line and as a JSON-serializable dict."""

    def __init__(self):
        self.lines = []
        self.events = []
        self.iteration = 0
        self.part = None

    def __call__(self, text, kind, **fields):
        self.lines.append(text)
        event = {"iteration": self.iteration, "part": self.part, "type": kind}
        event.update(fields)
        event["text"] = text.strip()
        self.events.append(event)


def _e(u, v):
    """Undirected edge as a sorted pair (for printing)."""
    return tuple(sorted((u, v)))


def _name(u, v):
    return f"{u}{v}"


def gabow_instr(G, *, skip_greedy_init=False, log=None):
    """The final algorithm with logging. ``log=None`` logs nothing."""
    L = log
    n = G.number_of_nodes()
    if n == 0:
        return set()

    adj = {v: [u for u in G[v] if u != v] for v in G}

    mate = {}
    if not skip_greedy_init:
        for v in G:
            if v in mate:
                continue
            for u in adj[v]:
                if u not in mate:
                    mate[v] = u
                    mate[u] = v
                    break

    class _UnionFind:
        __slots__ = ("parent", "rep", "size")

        def __init__(self, elements):
            self.parent = {v: v for v in elements}
            self.rep = dict(self.parent)
            self.size = dict.fromkeys(self.parent, 1)

        def _find_root(self, x):
            parent = self.parent
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def find(self, x):
            return self.rep[self._find_root(x)]

        def union(self, x, y):
            rx, ry = self._find_root(x), self._find_root(y)
            if rx == ry:
                return
            if self.size[rx] < self.size[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]

        def make_rep(self, x):
            self.rep[self._find_root(x)] = x

    class _BucketQueue:
        """FIFO buckets indexed by Delta. When logging, `meta` runs in
        parallel and records (inserted at Delta, scanned from) per entry."""

        __slots__ = ("buckets", "meta", "size")

        def __init__(self, size):
            self.size = size
            self.buckets = [deque() for _ in range(size)]
            self.meta = [deque() for _ in range(size)] if L else None

        def insert(self, edge, d, info=None):
            if d < self.size:
                self.buckets[d].append(edge)
                if self.meta is not None:
                    self.meta[d].append(info)

        def pop(self, d):
            if d >= self.size or not self.buckets[d]:
                return None
            return self.buckets[d].popleft()

        def pop_meta(self, d):
            return self.meta[d].popleft()

        def contents(self, d):
            if d >= self.size:
                return []
            return list(zip(self.buckets[d], self.meta[d]))

    def phase1():
        label = {v: ("EVEN" if v not in mate else "UNLABELED") for v in G}
        parent = {}
        bd = dict.fromkeys(G, 1)
        bDelta = dict.fromkeys(G, 0)
        source_bridge = {}
        target_bridge = {}
        base = _UnionFind(G)
        dbase = _UnionFind(G)
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
            if mate.get(u) == z or label[base.find(u)] == "ODD":
                return
            p = d(z) + d(u)
            if label[u] == "UNLABELED":
                queue.insert((z, u), Delta + p, (Delta, z))
            else:
                queue.insert((z, u), Delta + p // 2, (Delta, z))

        def shrink_path(b, x, y, dunions):
            v = base.find(x)
            while v != b:
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

        def log_level_end(committed):
            state = [
                {
                    "v": v,
                    "label": label[base.find(v)],
                    "d": d(v),
                    "base": base.find(v),
                }
                for v in sorted(T)
            ]
            L(
                f"    Delta={Delta}: end of level: "
                + (
                    ", ".join(
                        f"{s['v']}:{s['label']} d={s['d']} base={s['base']}"
                        for s in state
                    )
                    or "no free vertices (T is empty)"
                ),
                "level_end",
                Delta=Delta,
                vertices=state,
            )
            if committed:
                groups = {}
                for v in sorted(T):
                    groups.setdefault(dbase.find(v), []).append(v)
                L(
                    f"    Delta={Delta}: dbase after commit: "
                    + (", ".join(f"{r}:{m}" for r, m in sorted(groups.items())) or "-"),
                    "dbase",
                    Delta=Delta,
                    partition={str(r): m for r, m in sorted(groups.items())},
                )
            else:
                L(
                    f"    Delta={Delta}: augmenting path found at this level; "
                    "its unions are not committed to dbase",
                    "dbase_not_committed",
                    Delta=Delta,
                )

        for v in T:
            for u in adj[v]:
                scan_edge(u, v)

        found_sap = False
        dunions = []
        while 2 * Delta <= n:
            if L:
                entries = [
                    {"x": x, "y": y, "inserted_at": ins, "scanned_from": frm}
                    for (x, y), (ins, frm) in queue.contents(Delta)
                ]
                L(
                    f"    Delta={Delta}: bucket {Delta} = "
                    + str(
                        [
                            (e["x"], e["y"], e["inserted_at"], e["scanned_from"])
                            for e in entries
                        ]
                    ),
                    "bucket",
                    Delta=Delta,
                    entries=entries,
                )
            while True:
                e = queue.pop(Delta)
                if e is None:
                    break
                info = queue.pop_meta(Delta) if L else None
                x, y = e
                if label[base.find(x)] != "EVEN":
                    x, y = y, x
                bx, by = base.find(x), base.find(y)
                if y == mate.get(x) or bx == by or label[by] == "ODD":
                    if L:
                        reason = (
                            "matched edge"
                            if y == mate.get(x)
                            else ("same blossom" if bx == by else "other end ODD")
                        )
                        L(
                            f"    Delta={Delta}: pop {_name(x, y)}: skipped ({reason})",
                            "pop",
                            Delta=Delta,
                            x=x,
                            y=y,
                            inserted_at=info[0],
                            scanned_from=info[1],
                            outcome="skipped",
                            reason=reason,
                        )
                    continue
                if label[by] == "UNLABELED":
                    if L:
                        L(
                            f"    Delta={Delta}: pop {_name(x, y)}: grow",
                            "pop",
                            Delta=Delta,
                            x=x,
                            y=y,
                            inserted_at=info[0],
                            scanned_from=info[1],
                            outcome="grow",
                        )
                        L(
                            f"    Delta={Delta}: grow via {_name(x, y)}: {y} odd, {mate[y]} even",
                            "grow",
                            Delta=Delta,
                            x=x,
                            y=y,
                            odd=y,
                            even=mate[y],
                        )
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
                elif (
                    label[bx] == "EVEN"
                    and (bd[x] - (Delta - bDelta[x])) + (bd[y] - (Delta - bDelta[y]))
                    == 0
                ):
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
                        b = hy if path1.get(hy) == strue else hx
                        if L:
                            L(
                                f"    Delta={Delta}: pop {_name(x, y)}: blossom",
                                "pop",
                                Delta=Delta,
                                x=x,
                                y=y,
                                inserted_at=info[0],
                                scanned_from=info[1],
                                outcome="blossom",
                            )
                            L(
                                f"    Delta={Delta}: blossom via {_name(x, y)}, base {b}",
                                "blossom",
                                Delta=Delta,
                                x=x,
                                y=y,
                                base=b,
                            )
                        shrink_path(b, x, y, dunions)
                        shrink_path(b, y, x, dunions)
                    else:
                        if L:
                            L(
                                f"    Delta={Delta}: pop {_name(x, y)}: sap",
                                "pop",
                                Delta=Delta,
                                x=x,
                                y=y,
                                inserted_at=info[0],
                                scanned_from=info[1],
                                outcome="sap",
                            )
                            if not found_sap:
                                L(
                                    f"    Delta={Delta}: sap found via edge {_name(x, y)} "
                                    f"(length {2 * Delta - 1})",
                                    "sap",
                                    Delta=Delta,
                                    x=x,
                                    y=y,
                                    length=2 * Delta - 1,
                                )
                        found_sap = True
                elif L:
                    L(
                        f"    Delta={Delta}: pop {_name(x, y)}: skipped (not tight)",
                        "pop",
                        Delta=Delta,
                        x=x,
                        y=y,
                        inserted_at=info[0],
                        scanned_from=info[1],
                        outcome="skipped",
                        reason="not tight",
                    )
            if found_sap:
                if L:
                    log_level_end(committed=False)
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
            if L:
                log_level_end(committed=True)
            Delta += 1

        if not found_sap:
            if L:
                L("    Part I: no augmenting path", "no_sap")
            return None
        if L:
            duals = {v: d(v) for v in sorted(T)}
            L(
                f"    final duals: {duals}",
                "final_duals",
                Delta=Delta,
                duals={str(k): val for k, val in duals.items()},
            )

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
        if L:
            h_edges = sorted({_e(*e) for e in is_edge_of_H})
            L(
                f"    H edges: {h_edges}",
                "H",
                edges=[list(e) for e in h_edges],
                contracted_into={str(k): v for k, v in contracted_into.items()},
            )
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

    def phase2(H):
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
            before = len(mate_pairs)
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
            if L:
                added = mate_pairs[before:]
                L(
                    f"      trace_G({start_v}, {start_u}) adds {added}",
                    "trace_G",
                    start_v=start_v,
                    start_u=start_u,
                    pairs=[list(p) for p in added],
                )

        def augment(h_edges, root):
            mate_pairs = []
            for u, v in h_edges:
                mate_pairs.append((u, v))
                trace_G(u, rep[u], mate_pairs)
                trace_G(v, rep[v], mate_pairs)
            if L:
                # The augmenting path in G: new pairs plus the old matching
                # edges between their vertices, walked from the free endpoint
                # in the search root's blossom.
                verts = {x for p in mate_pairs for x in p}
                nbrs = {x: [] for x in verts}
                for a, b in mate_pairs:
                    nbrs[a].append(b)
                    nbrs[b].append(a)
                for a in verts:
                    b = mate.get(a)
                    if b in verts and a < b:
                        nbrs[a].append(b)
                        nbrs[b].append(a)
                ends = sorted(x for x in verts if len(nbrs[x]) == 1)
                start = next((x for x in ends if x in contracted_into[root]), ends[0])
                path, prev = [start], None
                while len(path) < len(verts):
                    nxt = next(y for y in nbrs[path[-1]] if y != prev)
                    prev = path[-1]
                    path.append(nxt)
                L(
                    f"    augment: mate pairs {mate_pairs}; path in G "
                    + ",".join(map(str, path)),
                    "augment",
                    pairs=[list(p) for p in mate_pairs],
                    path=path,
                )
            for a, b in mate_pairs:
                mate[a] = b
                mate[b] = a

        def log_stack(stack, why):
            frames = [f[0] for f in stack]
            L(
                f"      stack ({why}): {frames}",
                "stack",
                why=why,
                frames=frames,
            )

        augmentations = 0
        for root in list(contracted_into):
            if mateHG.get(root) is not None or labelHG.get(root) is not None:
                continue
            labelHG[root] = "EVEN"
            even_timeHG[root] = tG
            if L:
                L(f"    Part II: root {root}", "root", root=root)
                L(
                    f"      even_time[{root}] = {tG}",
                    "even_time",
                    vertex=root,
                    time=tG,
                )
            tG += 1

            stack = [[root, edge_iter_for(root)]]
            if L:
                log_stack(stack, f"push {root}")
            found_h_edges = None
            while stack:
                vh, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    if L:
                        log_stack(stack, f"pop {vh}")
                    continue
                v, u = nxt
                uh = rep[u]
                if mateHG.get(vh) == uh:
                    if L:
                        L(
                            f"    scan {_name(v, u)} from {vh}: ignored (matched edge)",
                            "scan",
                            vh=vh,
                            v=v,
                            u=u,
                            outcome="ignored",
                            reason="matched edge",
                        )
                    continue
                if labelHG.get(uh) is None:
                    parentHG[uh] = (v, u)
                    mateHG_uh = mateHG.get(uh)
                    if mateHG_uh is None:
                        labelHG[uh] = "ODD"
                        if L:
                            L(
                                f"    scan {_name(v, u)} from {vh}: {uh} free -> path",
                                "scan",
                                vh=vh,
                                v=v,
                                u=u,
                                outcome="free vertex -> path",
                            )
                        traced = trace_HG(rep[v], root)
                        if L:
                            L(
                                f"      trace_HG({rep[v]}, {root}) -> {traced}",
                                "trace_HG",
                                start=rep[v],
                                root=root,
                                edges=[list(e) for e in traced],
                            )
                        found_h_edges = [(v, u)] + traced
                        break
                    labelHG[uh] = "ODD"
                    labelHG[mateHG_uh] = "EVEN"
                    even_timeHG[mateHG_uh] = tG
                    if L:
                        L(
                            f"    scan {_name(v, u)} from {vh}: grow ({uh} odd, "
                            f"{mateHG_uh} even)",
                            "scan",
                            vh=vh,
                            v=v,
                            u=u,
                            outcome="grow",
                            odd=uh,
                            even=mateHG_uh,
                        )
                        L(
                            f"      even_time[{mateHG_uh}] = {tG}",
                            "even_time",
                            vertex=mateHG_uh,
                            time=tG,
                        )
                    tG += 1
                    stack.append([mateHG_uh, edge_iter_for(mateHG_uh)])
                    if L:
                        log_stack(stack, f"push {mateHG_uh}")
                else:
                    bh = dbase.find(vh)
                    zh = dbase.find(uh)
                    if bh != zh and even_timeHG.get(bh, -1) < even_timeHG.get(zh, -1):
                        tmp = []
                        endpoints_of_M = []
                        cur = zh
                        while cur != bh:
                            endpoints_of_M.append(cur)
                            cur = mateHG[cur]
                            endpoints_of_M.append(cur)
                            tmp.append(cur)
                            pv, pu = parentHG[cur]
                            other = pu if rep[pv] == cur else pv
                            cur = dbase.find(rep[other])
                        for node in endpoints_of_M:
                            dbase.union(node, bh)
                        dbase.make_rep(bh)
                        for node in tmp:
                            bridgeHG[node] = (v, u)
                        if L:
                            L(
                                f"    scan {_name(v, u)} from {vh}: blossom step, "
                                f"bh={bh}, zh={zh}, tmp (deep -> shallow) = {tmp}, "
                                f"pushed in this order (u_1 = {tmp[-1]} on top), "
                                f"bridge = {(v, u)} for {tmp}",
                                "scan",
                                vh=vh,
                                v=v,
                                u=u,
                                outcome="blossom step",
                                bh=bh,
                                zh=zh,
                                tmp=tmp,
                                pushed=tmp,
                                bridge=[v, u],
                            )
                        for node in tmp:
                            stack.append([node, edge_iter_for(node)])
                        if L:
                            log_stack(stack, f"push {tmp}")
                    elif L:
                        reason = (
                            "same blossom"
                            if bh == zh
                            else f"timestamp test failed (even_time[{bh}]="
                            f"{even_timeHG.get(bh, -1)} >= even_time[{zh}]="
                            f"{even_timeHG.get(zh, -1)})"
                        )
                        L(
                            f"    scan {_name(v, u)} from {vh}: ignored ({reason})",
                            "scan",
                            vh=vh,
                            v=v,
                            u=u,
                            outcome="ignored",
                            reason=reason,
                        )
            if found_h_edges is not None:
                if L:
                    L(
                        f"    Part II: root {root}: path edges "
                        f"{sorted(_e(*e) for e in found_h_edges)}",
                        "path",
                        root=root,
                        edges=[list(e) for e in found_h_edges],
                    )
                augment(found_h_edges, root)
                augmentations += 1

        return augmentations

    iteration = 0
    while True:
        iteration += 1
        if L:
            L.iteration = iteration
            L.part = None
            pairs = sorted(_e(a, b) for a, b in mate.items() if a < b)
            L(
                f"Iteration {iteration}: M = {pairs}",
                "iteration",
                M=[list(p) for p in pairs],
            )
            L.part = "I"
        H = phase1()
        if H is None:
            break
        if L:
            L.part = "II"
        gained = phase2(H)
        if gained == 0:
            raise nx.NetworkXAlgorithmError(
                "phase 1 found an augmenting path but phase 2 found none"
            )

    return matching_dict_to_set(mate)


# ---------------------------------------------------------------------------
# Examples and verification
# ---------------------------------------------------------------------------


def example_a():
    G = nx.Graph()
    G.add_nodes_from(range(1, 9))
    G.add_edges_from(
        sorted([(1, 2), (1, 7), (2, 3), (2, 5), (3, 4), (4, 6), (5, 6), (5, 8)])
    )
    return G


def example_b():
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 3), (2, 4)])
    return G


def run_example(name, G):
    log = Log()
    M = gabow_instr(G, skip_greedy_init=True, log=log)
    final = sorted(_e(*e) for e in M)
    log.iteration, log.part = None, None
    log(f"Final matching: {final}", "final", M=[list(e) for e in final])
    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / f"{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    (OUTDIR / f"{name}.json").write_text(
        json.dumps(
            {
                "example": name,
                "nodes": list(G.nodes()),
                "edges": [list(e) for e in G.edges()],
                "start": "empty matching",
                "events": log.events,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{name}: {len(log.events)} events -> {OUTDIR / (name + '.txt')}, .json")
    return log


def verify(n_graphs, seed=0):
    """Same matching as networkx.max_cardinality_matching_gabow, edge for
    edge, with logging off (and on, for every tenth graph)."""
    rng = random.Random(seed)
    checked = 0
    for i in range(n_graphs):
        n = rng.randint(2, 60)
        p = rng.choice([0.05, 0.1, 0.2, 0.4, 0.7])
        base = nx.gnm_random_graph(
            n, int(p * n * (n - 1) / 2), seed=rng.randrange(2**31)
        )
        nodes = list(base)
        order = rng.choice(["ascending", "descending", "random"])
        if order == "descending":
            nodes.reverse()
        elif order == "random":
            rng.shuffle(nodes)
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(base.edges())
        for skip in (False, True):
            ref = nx.max_cardinality_matching_gabow(G, _skip_greedy_init=skip)
            got = gabow_instr(G, skip_greedy_init=skip)
            if {_e(*e) for e in got} != {_e(*e) for e in ref}:
                sys.exit(f"MISMATCH on graph {i} (skip_greedy_init={skip})")
            if i % 10 == 0:
                got_logged = gabow_instr(G, skip_greedy_init=skip, log=Log())
                if {_e(*e) for e in got_logged} != {_e(*e) for e in ref}:
                    sys.exit(f"MISMATCH with logging on, graph {i} (skip={skip})")
            checked += 1
    print(
        f"verify: {n_graphs} graphs x 2 starts = {checked} runs, identical "
        f"matchings to networkx.max_cardinality_matching_gabow (logging off; "
        f"logging also on for every tenth graph)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", type=int, metavar="N", help="check N random graphs")
    args = p.parse_args()
    if args.verify:
        verify(args.verify)
        return
    run_example("example_A", example_a())
    run_example("example_B", example_b())


if __name__ == "__main__":
    main()
