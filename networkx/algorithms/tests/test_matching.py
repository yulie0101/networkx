import math
from itertools import permutations

import pytest

import networkx as nx
from networkx.utils import edges_equal


@pytest.mark.parametrize(
    "fn", (nx.is_matching, nx.is_maximal_matching, nx.is_perfect_matching)
)
@pytest.mark.parametrize(
    "edgeset",
    (
        {(0, 5)},  # Single edge, node not in G
        {(5, 0)},  # for both edge orders
        {(0, 5), (2, 3)},  # node not in G, but other edge is valid matching
        {(5, 5), (2, 3)},  # Self-loop hits node not in G validation first
    ),
)
def test_is_matching_node_not_in_G(fn, edgeset):
    """All is_*matching functions have consistent exception message for node
    not in G."""
    G = nx.path_graph(4)
    with pytest.raises(nx.NetworkXError, match="matching.*with node not in G"):
        fn(G, edgeset)


@pytest.mark.parametrize(
    "fn", (nx.is_matching, nx.is_maximal_matching, nx.is_perfect_matching)
)
@pytest.mark.parametrize(
    "edgeset",
    (
        {(0, 1, 2), (2, 3)},  # 3-tuple
        {(0,), (2, 3)},  # 1-tuple
    ),
)
def test_is_matching_invalid_edge(fn, edgeset):
    """All is_*matching functions have consistent exception message for invalid
    edges in matching."""
    G = nx.path_graph(4)
    with pytest.raises(nx.NetworkXError, match=".*non-2-tuple edge.*"):
        fn(G, edgeset)


@pytest.mark.parametrize("graph_type", (nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph))
@pytest.mark.parametrize(
    "fn",
    (
        nx.max_weight_matching,
        nx.min_weight_matching,
        nx.maximal_matching,
        nx.max_cardinality_matching_gabow,
    ),
)
def test_wrong_graph_type(fn, graph_type):
    G = graph_type()
    with pytest.raises(nx.NetworkXNotImplemented):
        fn(G)


def _brute_force_max_matching_size(G):
    """Independent oracle for small graphs: exhaustively search for the
    largest matching by recursive backtracking (pick an unmatched vertex,
    either leave it unmatched or pair it with each available neighbor).
    Only intended for graphs with few nodes -- this is exponential time.
    """
    nodes = list(G.nodes())
    adj = {v: {u for u in G[v] if u != v} for v in nodes}
    best = 0

    def backtrack(remaining, size):
        nonlocal best
        if size > best:
            best = size
        if not remaining:
            return
        v = next(iter(remaining))
        rest = remaining - {v}
        # Leave v unmatched.
        backtrack(rest, size)
        # Try matching v to each still-available neighbor.
        for u in adj[v] & remaining:
            backtrack(rest - {u}, size + 1)

    backtrack(frozenset(nodes), 0)
    return best


class TestMaxWeightMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.max_weight_matching` function.

    """

    def test_trivial1(self):
        """Empty graph"""
        G = nx.Graph()
        assert nx.max_weight_matching(G) == set()
        assert nx.min_weight_matching(G) == set()

    def test_selfloop(self):
        G = nx.Graph()
        G.add_edge(0, 0, weight=100)
        assert nx.max_weight_matching(G) == set()
        assert nx.min_weight_matching(G) == set()

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert edges_equal(nx.max_weight_matching(G), {(0, 1)})
        assert edges_equal(nx.min_weight_matching(G), {(0, 1)})

    def test_two_path(self):
        G = nx.Graph()
        G.add_edge("one", "two", weight=10)
        G.add_edge("two", "three", weight=11)
        assert edges_equal(nx.max_weight_matching(G), {("two", "three")})
        assert edges_equal(nx.min_weight_matching(G), {("one", "two")})

    def test_path(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=11)
        G.add_edge(3, 4, weight=5)
        assert edges_equal(nx.max_weight_matching(G), {(2, 3)})
        assert edges_equal(nx.max_weight_matching(G, weight=None), {(1, 2), (3, 4)})
        assert edges_equal(nx.min_weight_matching(G), {(1, 2), (3, 4)})
        assert edges_equal(nx.min_weight_matching(G, weight=None), {(1, 2), (3, 4)})

    def test_square(self):
        G = nx.Graph()
        G.add_edge(1, 4, weight=2)
        G.add_edge(2, 3, weight=2)
        G.add_edge(1, 2, weight=1)
        G.add_edge(3, 4, weight=4)
        assert edges_equal(nx.max_weight_matching(G), {(1, 2), (3, 4)})
        assert edges_equal(nx.min_weight_matching(G), {(1, 4), (2, 3)})

    def test_edge_attribute_name(self):
        G = nx.Graph()
        G.add_edge("one", "two", weight=10, abcd=11)
        G.add_edge("two", "three", weight=11, abcd=10)
        assert edges_equal(nx.max_weight_matching(G, weight="abcd"), {("one", "two")})
        assert edges_equal(nx.min_weight_matching(G, weight="abcd"), {("two", "three")})

    def test_floating_point_weights(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=math.pi)
        G.add_edge(2, 3, weight=math.exp(1))
        G.add_edge(1, 3, weight=3.0)
        G.add_edge(1, 4, weight=math.sqrt(2.0))
        assert edges_equal(nx.max_weight_matching(G), {(1, 4), (2, 3)})
        assert edges_equal(nx.min_weight_matching(G), {(1, 4), (2, 3)})

    def test_negative_weights(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=-2)
        G.add_edge(2, 3, weight=1)
        G.add_edge(2, 4, weight=-1)
        G.add_edge(3, 4, weight=-6)
        assert edges_equal(nx.max_weight_matching(G), {(1, 2)})
        assert edges_equal(
            nx.max_weight_matching(G, maxcardinality=True), {(1, 3), (2, 4)}
        )
        assert edges_equal(nx.min_weight_matching(G), {(1, 2), (3, 4)})

    def test_s_blossom(self):
        """Create S-blossom and use it for augmentation:"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)])
        answer = {(1, 2), (3, 4)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

        G.add_weighted_edges_from([(1, 6, 5), (4, 5, 6)])
        answer = {(1, 6), (2, 3), (4, 5)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_s_t_blossom(self):
        """Create S-blossom, relabel as T-blossom, use for augmentation:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)]
        )
        answer = {(1, 6), (2, 3), (4, 5)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

        G.add_edge(4, 5, weight=3)
        G.add_edge(1, 6, weight=4)
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

        G.remove_edge(1, 6)
        G.add_edge(3, 6, weight=4)
        answer = {(1, 2), (3, 6), (4, 5)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom(self):
        """Create nested S-blossom, use for augmentation:"""

        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 9),
                (1, 3, 9),
                (2, 3, 10),
                (2, 4, 8),
                (3, 5, 8),
                (4, 5, 10),
                (5, 6, 6),
            ]
        )
        expected_edgeset = {(1, 3), (2, 4), (5, 6)}
        expected = {frozenset(e) for e in expected_edgeset}
        answer = {frozenset(e) for e in nx.max_weight_matching(G)}
        assert answer == expected
        answer = {frozenset(e) for e in nx.min_weight_matching(G)}
        assert answer == expected

    def test_nested_s_blossom_relabel(self):
        """Create S-blossom, relabel as S, include in nested S-blossom:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 10),
                (1, 7, 10),
                (2, 3, 12),
                (3, 4, 20),
                (3, 5, 20),
                (4, 5, 25),
                (5, 6, 10),
                (6, 7, 10),
                (7, 8, 8),
            ]
        )
        answer = {(1, 2), (3, 4), (5, 6), (7, 8)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom_expand(self):
        """Create nested S-blossom, augment, expand recursively:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 8),
                (1, 3, 8),
                (2, 3, 10),
                (2, 4, 12),
                (3, 5, 12),
                (4, 5, 14),
                (4, 6, 12),
                (5, 7, 12),
                (6, 7, 14),
                (7, 8, 12),
            ]
        )
        answer = {(1, 2), (3, 5), (4, 6), (7, 8)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_s_blossom_relabel_expand(self):
        """Create S-blossom, relabel as T, expand:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 23),
                (1, 5, 22),
                (1, 6, 15),
                (2, 3, 25),
                (3, 4, 22),
                (4, 5, 25),
                (4, 8, 14),
                (5, 7, 13),
            ]
        )
        answer = {(1, 6), (2, 3), (4, 8), (5, 7)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nested_s_blossom_relabel_expand(self):
        """Create nested S-blossom, relabel as T, expand:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 19),
                (1, 3, 20),
                (1, 8, 8),
                (2, 3, 25),
                (2, 4, 18),
                (3, 5, 18),
                (4, 5, 13),
                (4, 7, 7),
                (5, 6, 7),
            ]
        )
        answer = {(1, 8), (2, 3), (4, 7), (5, 6)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom1(self):
        """Create blossom, relabel as T in more than one way, expand,
        augment:
        """
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 35),
                (5, 7, 26),
                (9, 10, 5),
            ]
        )
        answer = {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom2(self):
        """Again but slightly different:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 26),
                (5, 7, 40),
                (9, 10, 5),
            ]
        )
        answer = {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_least_slack(self):
        """Create blossom, relabel as T, expand such that a new
        least-slack S-to-free dge is produced, augment:
        """
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 28),
                (5, 7, 26),
                (9, 10, 5),
            ]
        )
        answer = {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_augmenting(self):
        """Create nested blossom, relabel as T in more than one way"""
        # expand outer blossom such that inner blossom ends up on an
        # augmenting path:
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 45),
                (1, 7, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 95),
                (4, 6, 94),
                (5, 6, 94),
                (6, 7, 50),
                (1, 8, 30),
                (3, 11, 35),
                (5, 9, 36),
                (7, 10, 26),
                (11, 12, 5),
            ]
        )
        answer = {(1, 8), (2, 3), (4, 6), (5, 9), (7, 10), (11, 12)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_nasty_blossom_expand_recursively(self):
        """Create nested S-blossom, relabel as S, expand recursively:"""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (1, 2, 40),
                (1, 3, 40),
                (2, 3, 60),
                (2, 4, 55),
                (3, 5, 55),
                (4, 5, 50),
                (1, 8, 15),
                (5, 7, 30),
                (7, 6, 10),
                (8, 10, 10),
                (4, 9, 30),
            ]
        )
        answer = {(1, 2), (3, 5), (4, 9), (6, 7), (8, 10)}
        assert edges_equal(nx.max_weight_matching(G), answer)
        assert edges_equal(nx.min_weight_matching(G), answer)

    def test_min_weight_matching_max_cardinality(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 1000), (2, 3, 2), (3, 4, 3000)])
        # The minimum-weight maximal matching is {(2, 3)}; the minimum-weight
        # maximum-cardinality matching is {(1, 2), (3, 4)}. See gh-8062.
        answer = {(1, 2), (3, 4)}
        assert edges_equal(nx.min_weight_matching(G), answer)


class TestIsMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_empty_matching(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, set())

    def test_single_edge(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(1, 2)})

    def test_edge_order(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(0, 1), (2, 3)})
        assert nx.is_matching(G, {(1, 0), (2, 3)})
        assert nx.is_matching(G, {(0, 1), (3, 2)})
        assert nx.is_matching(G, {(1, 0), (3, 2)})

    def test_valid_matching(self):
        G = nx.path_graph(4)
        assert nx.is_matching(G, {(0, 1), (2, 3)})

    def test_selfloops(self):
        G = nx.path_graph(4)
        # selfloop edge not in G
        assert not nx.is_matching(G, {(0, 0), (1, 2), (2, 3)})
        # selfloop edge in G
        G.add_edge(0, 0)
        assert not nx.is_matching(G, {(0, 0), (1, 2)})

    def test_invalid_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_matching(G, {(0, 1), (1, 2), (2, 3)})

    def test_invalid_edge(self):
        G = nx.path_graph(4)
        assert not nx.is_matching(G, {(0, 3), (1, 2)})

        G = nx.DiGraph(G.edges)
        assert nx.is_matching(G, {(0, 1)})
        assert not nx.is_matching(G, {(1, 0)})


class TestIsMaximalMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_maximal_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_maximal_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_valid(self):
        G = nx.path_graph(4)
        assert nx.is_maximal_matching(G, {(0, 1), (2, 3)})

    def test_not_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_maximal_matching(G, {(0, 1), (1, 2), (2, 3)})
        assert not nx.is_maximal_matching(G, {(0, 3)})
        G.add_edge(0, 0)
        assert not nx.is_maximal_matching(G, {(0, 0)})

    def test_not_maximal(self):
        G = nx.path_graph(4)
        assert not nx.is_maximal_matching(G, {(0, 1)})


class TestIsPerfectMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_perfect_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_perfect_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_valid(self):
        G = nx.path_graph(4)
        assert nx.is_perfect_matching(G, {(0, 1), (2, 3)})

    def test_valid_not_path(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 4)
        G.add_edge(1, 4)
        G.add_edge(5, 2)

        assert nx.is_perfect_matching(G, {(1, 4), (0, 3), (5, 2)})

    def test_selfloops(self):
        G = nx.path_graph(4)
        # selfloop edge not in G
        assert not nx.is_perfect_matching(G, {(0, 0), (1, 2), (2, 3)})
        # selfloop edge in G
        G.add_edge(0, 0)
        assert not nx.is_perfect_matching(G, {(0, 0), (1, 2)})

    def test_not_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_perfect_matching(G, {(0, 3)})
        assert not nx.is_perfect_matching(G, {(0, 1), (1, 2), (2, 3)})

    def test_maximal_but_not_perfect(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 4)
        G.add_edge(1, 4)

        assert not nx.is_perfect_matching(G, {(1, 4), (0, 3)})


class TestMaximalMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.maximal_matching`.

    """

    def test_valid_matching(self):
        edges = [(1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (3, 6), (5, 6)]
        G = nx.Graph(edges)
        matching = nx.maximal_matching(G)
        assert nx.is_maximal_matching(G, matching)

    def test_single_edge_matching(self):
        # In the star graph, any maximal matching has just one edge.
        G = nx.star_graph(5)
        matching = nx.maximal_matching(G)
        assert 1 == len(matching)
        assert nx.is_maximal_matching(G, matching)

    def test_self_loops(self):
        # Create the path graph with two self-loops.
        G = nx.path_graph(3)
        G.add_edges_from([(0, 0), (1, 1)])
        matching = nx.maximal_matching(G)
        assert len(matching) == 1
        # The matching should never include self-loops.
        assert not any(u == v for u, v in matching)
        assert nx.is_maximal_matching(G, matching)

    def test_ordering(self):
        """Tests that a maximal matching is computed correctly
        regardless of the order in which nodes are added to the graph.

        """
        for nodes in permutations(range(3)):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from([(0, 1), (0, 2)])
            matching = nx.maximal_matching(G)
            assert len(matching) == 1
            assert nx.is_maximal_matching(G, matching)


class TestMaxCardinalityMatchingGabow:
    """Unit tests for
    :func:`~networkx.algorithms.matching.max_cardinality_matching_gabow`.
    """

    def _check(self, G, expected_size=None):
        """Check both use_heuristic_fallback settings against each other,
        against the size found by max_weight_matching(maxcardinality=True),
        and (for small graphs) against a brute-force oracle -- and confirm
        the result is always a valid matching of G.
        """
        got = nx.max_cardinality_matching_gabow(G)
        got_heur = nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=True)
        assert nx.is_matching(G, got)
        assert nx.is_matching(G, got_heur)
        assert len(got) == len(got_heur)
        ref = nx.max_weight_matching(G, maxcardinality=True)
        assert len(got) == len(ref)
        if G.number_of_nodes() <= 10:
            assert len(got) == _brute_force_max_matching_size(G)
        if expected_size is not None:
            assert len(got) == expected_size
        return got

    # -- trivial / edge cases --------------------------------------------

    def test_no_nodes(self):
        G = nx.Graph()
        assert nx.max_cardinality_matching_gabow(G) == set()

    def test_single_node_no_edges(self):
        G = nx.Graph()
        G.add_node(0)
        self._check(G, expected_size=0)

    def test_no_edges(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        self._check(G, expected_size=0)

    def test_single_edge(self):
        G = nx.Graph([(0, 1)])
        assert edges_equal(self._check(G, expected_size=1), {(0, 1)})

    def test_self_loops_are_ignored(self):
        G = nx.Graph()
        G.add_edges_from([(0, 0), (0, 1), (1, 1), (2, 2)])
        got = self._check(G, expected_size=1)
        assert not any(u == v for u, v in got)

    def test_disconnected_components(self):
        G = nx.disjoint_union_all(
            [nx.cycle_graph(5), nx.complete_graph(4), nx.path_graph(3), nx.Graph([(0, 0)])]
        )
        # C5 contributes 2, K4 contributes 2, P3 contributes 1, the
        # self-loop-only component contributes 0.
        self._check(G, expected_size=5)

    def test_isolated_vertices_mixed_in(self):
        G = nx.path_graph(4)
        G.add_nodes_from(["isolated_a", "isolated_b"])
        self._check(G, expected_size=2)

    # -- hand-verified small graphs ---------------------------------------

    def test_hand_verified_square(self):
        # A 4-cycle has a perfect matching of size 2.
        G = nx.cycle_graph(4)
        assert edges_equal(self._check(G, expected_size=2), {(0, 1), (2, 3)})

    def test_hand_verified_star(self):
        # In a star, only one edge can ever be matched.
        G = nx.star_graph(5)
        self._check(G, expected_size=1)

    def test_hand_verified_path(self):
        # P_6 (6 nodes, 5 edges) has a perfect matching of size 3.
        G = nx.path_graph(6)
        self._check(G, expected_size=3)

    def test_hand_verified_two_triangles_sharing_a_bridge(self):
        # Two odd triangles {0,1,2} and {3,4,5} joined by bridge 2-3.
        # A perfect matching of size 3 exists by using the bridge itself:
        # (0,1), (2,3), (4,5).
        G = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
        self._check(G, expected_size=3)

    # -- blossom contraction (odd cycles) ----------------------------------

    @pytest.mark.parametrize("k", [3, 5, 7, 9, 11])
    def test_odd_cycle(self, k):
        G = nx.cycle_graph(k)
        self._check(G, expected_size=(k - 1) // 2)

    def test_odd_cycle_with_pendant(self):
        # A blossom (C5) with an extra pendant vertex attached to it: the
        # augmenting path must pass straight through the blossom to reach
        # the pendant, forcing the blossom to actually be contracted and
        # then correctly "un-contracted" during path reconstruction.
        G = nx.cycle_graph(5)
        G.add_edge(0, "pendant")
        self._check(G, expected_size=3)

    def test_chained_blossoms(self):
        # A chain of k triangles, each sharing a vertex with the next,
        # forcing repeated blossom shrinks along a single search.
        G = nx.Graph()
        for i in range(6):
            base = i * 2
            G.add_edges_from([(base, base + 1), (base + 1, base + 2), (base + 2, base)])
        self._check(G)

    def test_nested_style_blossom(self):
        # The classic "blossom in a blossom" stress graph: an outer 5-cycle
        # where one edge is replaced by a path that itself closes into a
        # smaller odd cycle, forcing a blossom step to fire while already
        # inside another blossom's search.
        G = nx.Graph(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                (2, 5),
                (5, 6),
                (6, 3),
            ]
        )
        self._check(G)

    # -- cross-validation on random graphs ----------------------------------

    @pytest.mark.parametrize("seed", range(25))
    def test_random_small_graphs(self, seed):
        n = (seed % 9) + 1
        p = [0.1, 0.3, 0.5, 0.7, 0.9][seed % 5]
        G = nx.gnp_random_graph(n, p, seed=seed)
        self._check(G)

    @pytest.mark.parametrize("seed", range(15))
    def test_random_larger_graphs(self, seed):
        n = 20 + 5 * (seed % 6)
        p = [0.05, 0.15, 0.4][seed % 3]
        G = nx.gnp_random_graph(n, p, seed=100 + seed)
        self._check(G)

    def test_random_graphs_with_self_loops(self):
        rng_seeds = range(10)
        for seed in rng_seeds:
            G = nx.gnp_random_graph(12, 0.3, seed=seed)
            nodes = list(G.nodes())
            if nodes:
                G.add_edge(nodes[seed % len(nodes)], nodes[seed % len(nodes)])
            self._check(G)

    # -- heuristic fallback --------------------------------------------------

    @pytest.mark.parametrize(
        "G",
        [
            nx.Graph(),
            nx.path_graph(1),
            nx.cycle_graph(7),
            nx.complete_graph(9),
            nx.star_graph(6),
            nx.disjoint_union_all([nx.cycle_graph(5), nx.cycle_graph(5)]),
        ],
    )
    def test_heuristic_fallback_matches_default(self, G):
        default = nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=False)
        heuristic = nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=True)
        assert nx.is_matching(G, default)
        assert nx.is_matching(G, heuristic)
        assert len(default) == len(heuristic)

    @pytest.mark.parametrize("seed", range(20))
    def test_heuristic_fallback_matches_random(self, seed):
        n = 15 + seed
        p = 0.1 + 0.03 * (seed % 10)
        G = nx.gnp_random_graph(n, p, seed=seed)
        default = nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=False)
        heuristic = nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=True)
        assert len(default) == len(heuristic)
        assert len(default) == len(nx.max_weight_matching(G, maxcardinality=True))
