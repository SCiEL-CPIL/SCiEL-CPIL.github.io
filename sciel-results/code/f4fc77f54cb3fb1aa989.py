"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import defaultdict

P = {'n': 18, 'edge_order': [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [6, 14, 17], [11, 12, 16], [3, 10, 17], [10, 12, 14], [3, 9, 13], [0, 7, 13], [3, 14, 16], [1, 5, 8], [0, 9, 12], [9, 13, 17], [4, 11, 17], [4, 8, 13], [3, 14, 17], [3, 11, 15], [11, 13, 17], [8, 11, 12], [4, 9, 13], [2, 7, 9], [3, 6, 14], [0, 3, 5], [5, 7, 16], [2, 9, 16], [6, 7, 9], [1, 7, 8], [3, 10, 13], [3, 10, 14], [1, 10, 15], [2, 4, 14], [4, 11, 14], [6, 7, 10], [5, 6, 12], [3, 13, 17], [0, 5, 7], [5, 7, 11], [0, 2, 11], [9, 15, 16], [0, 2, 14], [5, 15, 17], [0, 3, 7], [10, 12, 15], [7, 9, 11], [4, 7, 17], [4, 11, 16], [3, 6, 17], [7, 11, 14], [1, 3, 8], [1, 10, 16], [3, 9, 16], [7, 8, 11], [7, 10, 14], [4, 10, 14], [3, 14, 15], [4, 14, 17], [4, 16, 17], [9, 12, 15], [0, 2, 4], [7, 9, 15], [10, 12, 17], [7, 8, 13], [8, 11, 15], [8, 16, 17], [5, 11, 13], [1, 11, 14], [0, 6, 10], [0, 8, 14], [5, 6, 16], [0, 6, 17], [1, 16, 17], [7, 11, 16], [2, 11, 16], [0, 1, 13], [6, 7, 14], [4, 7, 11], [6, 11, 14], [1, 7, 14], [1, 4, 8], [5, 9, 15], [8, 9, 15], [3, 11, 13], [2, 11, 14], [3, 12, 13], [2, 4, 8], [12, 14, 15], [1, 12, 17], [1, 14, 17], [6, 15, 16], [0, 11, 13], [2, 5, 15], [11, 12, 17], [5, 15, 16], [2, 6, 15], [5, 6, 15], [5, 8, 15], [0, 7, 9], [1, 4, 14], [9, 14, 15], [2, 4, 10], [3, 7, 15], [5, 7, 17], [1, 10, 14], [6, 8, 17], [1, 2, 15], [1, 14, 15], [2, 5, 17], [7, 9, 16], [10, 13, 17], [5, 9, 14], [0, 12, 13], [0, 13, 14], [7, 10, 12], [2, 11, 15], [4, 5, 10], [10, 16, 17], [4, 14, 16], [1, 8, 15], [5, 6, 14], [7, 11, 15], [3, 11, 12], [7, 9, 17], [4, 7, 14], [1, 5, 15], [3, 7, 12], [4, 5, 12], [3, 4, 15], [1, 6, 10], [1, 8, 10], [6, 10, 13], [9, 11, 17], [10, 14, 15], [2, 3, 5], [0, 5, 6], [4, 11, 12], [2, 6, 16], [2, 8, 10], [2, 3, 11], [9, 10, 11], [2, 9, 17], [0, 12, 16], [0, 6, 13], [1, 9, 13], [5, 6, 9], [4, 5, 17], [2, 7, 10], [2, 3, 7], [1, 7, 11], [3, 7, 17], [0, 2, 7], [3, 5, 10], [4, 7, 15], [1, 2, 11], [6, 8, 10], [2, 4, 6], [0, 2, 13], [2, 3, 17], [3, 7, 16], [3, 4, 10], [1, 6, 7], [6, 10, 12], [4, 6, 7], [7, 10, 17], [8, 13, 17], [1, 12, 15], [0, 8, 15], [2, 5, 7], [1, 3, 15], [2, 4, 11], [4, 6, 8], [9, 14, 16], [0, 13, 16], [0, 9, 16], [8, 10, 14], [3, 5, 12], [2, 9, 13], [2, 7, 12], [0, 1, 11], [1, 7, 13], [1, 4, 7], [0, 9, 17], [0, 1, 14], [8, 13, 14], [0, 2, 5], [1, 10, 12], [8, 12, 13], [1, 14, 16], [3, 8, 12], [5, 7, 14], [1, 5, 9], [5, 13, 16], [0, 1, 15], [0, 16, 17], [3, 5, 6], [5, 8, 11], [0, 3, 13], [0, 12, 14], [5, 6, 8], [5, 7, 12], [4, 5, 8], [0, 10, 11], [1, 8, 11], [1, 13, 16], [3, 12, 16], [3, 6, 8], [0, 2, 17], [9, 10, 12], [2, 5, 8], [1, 12, 14], [8, 9, 13], [0, 7, 8], [0, 14, 17], [4, 10, 13], [9, 12, 13], [5, 11, 15], [13, 14, 15], [1, 9, 10], [2, 7, 15], [6, 9, 10], [0, 12, 15], [12, 13, 16], [0, 9, 14], [6, 13, 15], [1, 4, 10], [4, 7, 16], [12, 15, 17], [2, 7, 14], [0, 4, 5], [3, 16, 17], [13, 15, 16], [5, 7, 10], [6, 7, 16], [6, 15, 17], [6, 12, 13], [1, 13, 15], [0, 4, 8], [7, 13, 15], [4, 9, 11], [6, 7, 13], [0, 8, 16], [5, 6, 11], [2, 4, 17], [3, 13, 14], [10, 11, 13], [2, 11, 12], [6, 7, 17], [4, 13, 15], [3, 5, 13], [0, 9, 15], [3, 13, 16], [4, 12, 15], [12, 16, 17], [2, 6, 12], [7, 8, 12], [0, 7, 17], [2, 10, 12], [8, 9, 16], [8, 10, 13], [6, 9, 16], [6, 13, 17], [1, 3, 13], [6, 11, 12], [6, 13, 14], [2, 4, 9], [0, 10, 15], [0, 14, 16], [1, 9, 17], [4, 8, 10], [3, 6, 13], [4, 6, 17], [1, 2, 8], [2, 10, 17], [2, 8, 16], [0, 7, 16], [0, 10, 17], [2, 14, 15], [6, 7, 11], [10, 11, 16], [6, 9, 11], [1, 8, 17], [1, 7, 9], [0, 6, 9], [10, 14, 17], [2, 5, 10], [4, 8, 9], [7, 8, 15], [2, 12, 17], [9, 10, 14], [8, 12, 17], [6, 7, 12], [5, 8, 14], [6, 9, 14], [5, 12, 14], [3, 15, 17], [10, 13, 16], [8, 14, 15], [7, 11, 17], [5, 10, 13], [5, 11, 14], [6, 12, 15], [2, 10, 15], [0, 5, 15], [12, 14, 17], [11, 13, 16], [2, 13, 16], [2, 9, 15], [0, 15, 17], [0, 6, 11], [2, 8, 13], [0, 1, 4], [4, 12, 13], [5, 9, 12], [0, 9, 10], [1, 9, 11], [7, 15, 16], [0, 1, 8], [5, 7, 9], [0, 1, 9], [11, 12, 14], [1, 8, 14], [6, 8, 12], [5, 8, 17], [3, 8, 15], [3, 5, 17], [2, 5, 11], [0, 3, 12], [5, 14, 16], [7, 14, 17], [6, 9, 12], [9, 10, 15], [7, 9, 13], [9, 11, 16], [3, 7, 11], [3, 9, 12], [3, 4, 7], [1, 5, 17], [0, 1, 3], [5, 10, 12], [2, 3, 8], [7, 13, 14], [0, 7, 12], [9, 13, 15], [3, 11, 14], [0, 4, 16], [12, 13, 15], [7, 9, 14], [2, 10, 16], [7, 9, 10], [3, 10, 12], [14, 15, 16], [5, 12, 16], [1, 3, 10], [4, 10, 15], [5, 6, 13], [1, 7, 10], [6, 9, 13], [2, 9, 11], [1, 7, 15], [1, 7, 16], [3, 10, 16], [1, 6, 13], [0, 9, 13], [5, 8, 13], [11, 13, 15], [2, 4, 15], [10, 11, 14], [13, 16, 17], [3, 5, 7], [0, 8, 13], [3, 8, 17], [2, 5, 6], [3, 4, 6], [0, 1, 10], [3, 8, 10], [0, 3, 8], [2, 13, 17], [0, 4, 15], [2, 7, 13], [3, 12, 17], [8, 10, 11], [4, 8, 12], [10, 12, 13], [11, 12, 15], [0, 4, 10], [8, 10, 17], [4, 5, 11], [0, 10, 16], [0, 6, 7], [3, 10, 11], [1, 15, 17], [3, 8, 13], [3, 6, 9], [4, 15, 17], [2, 4, 13], [4, 8, 11], [1, 10, 17], [0, 2, 12], [0, 7, 10], [0, 1, 16], [0, 7, 14], [4, 5, 6], [4, 5, 9], [2, 6, 10], [2, 6, 17], [0, 7, 15], [4, 5, 14], [3, 7, 10], [2, 16, 17], [5, 8, 12], [6, 11, 16], [6, 10, 15], [2, 15, 17], [4, 6, 14], [0, 10, 12], [10, 15, 17], [1, 6, 16], [0, 12, 17], [6, 11, 13], [1, 3, 9], [7, 12, 16], [3, 5, 14], [6, 10, 11], [3, 11, 16], [1, 4, 12], [7, 15, 17], [4, 9, 17], [7, 8, 17], [1, 8, 13], [3, 6, 12], [1, 13, 17], [2, 9, 10], [7, 9, 12], [1, 8, 9], [2, 8, 17], [0, 11, 14], [3, 6, 11], [2, 5, 12], [11, 15, 16], [1, 9, 12], [2, 7, 16], [5, 8, 10], [2, 6, 14], [9, 13, 14], [5, 9, 10], [2, 12, 14], [0, 8, 11], [8, 12, 16], [3, 4, 8], [7, 11, 13], [5, 14, 15], [2, 4, 16], [1, 2, 10], [11, 14, 15], [6, 8, 15], [10, 11, 15], [6, 8, 16], [11, 12, 13], [8, 11, 17], [11, 15, 17], [2, 3, 13], [1, 5, 11], [0, 5, 16], [4, 6, 15], [14, 16, 17], [0, 9, 11], [0, 3, 10], [0, 2, 3], [3, 10, 15], [10, 12, 16], [2, 11, 13], [4, 5, 7], [10, 15, 16], [8, 14, 17], [1, 13, 14], [2, 6, 8], [1, 2, 17], [2, 4, 5], [9, 13, 16], [4, 10, 12], [7, 14, 16], [5, 7, 13], [3, 4, 16], [2, 12, 15], [0, 3, 4], [6, 7, 15], [8, 11, 14], [5, 9, 13], [1, 2, 9], [0, 8, 12], [1, 5, 16], [0, 5, 14], [8, 10, 16], [1, 10, 13], [10, 11, 12], [4, 15, 16], [5, 13, 15], [1, 10, 11], [13, 14, 16], [2, 10, 14], [1, 5, 7], [6, 14, 16], [4, 6, 11], [2, 3, 12], [7, 14, 15], [7, 10, 16], [4, 12, 17], [0, 8, 10], [0, 7, 11], [1, 9, 14], [8, 13, 16], [4, 13, 16], [1, 8, 12], [1, 6, 14], [9, 11, 14], [8, 10, 12], [1, 3, 5], [0, 3, 14], [0, 4, 7], [2, 13, 14], [5, 11, 17], [1, 2, 16], [3, 7, 8], [2, 11, 17], [5, 13, 14], [3, 5, 11], [4, 8, 14], [1, 3, 17], [0, 14, 15], [5, 8, 9], [0, 15, 16], [0, 11, 12], [0, 2, 16], [2, 3, 9], [2, 5, 13], [8, 12, 15], [9, 10, 13], [7, 10, 13], [0, 6, 12], [3, 11, 17], [1, 5, 13], [3, 7, 9], [0, 1, 2], [1, 12, 13], [4, 9, 10], [1, 4, 6], [8, 9, 11], [3, 5, 15], [2, 14, 17], [2, 9, 14], [2, 8, 11], [5, 12, 13], [7, 10, 11], [9, 12, 16], [5, 12, 17], [0, 5, 10], [10, 14, 16], [5, 10, 15], [8, 9, 17], [1, 6, 15], [1, 8, 16], [6, 13, 16], [4, 5, 15], [8, 13, 15], [6, 10, 14], [3, 7, 13], [3, 4, 5], [0, 2, 9], [5, 6, 7], [8, 15, 17], [0, 11, 16], [5, 7, 15], [7, 12, 15], [6, 8, 9], [0, 1, 12], [6, 8, 11], [0, 1, 7], [0, 5, 8], [0, 13, 15], [0, 3, 11], [0, 4, 17], [9, 15, 17], [5, 11, 16], [9, 16, 17], [4, 11, 15], [8, 12, 14], [3, 5, 16], [3, 5, 8], [0, 6, 16], [2, 3, 14], [4, 6, 12], [4, 12, 16], [6, 8, 13], [5, 9, 17], [5, 9, 11], [3, 7, 14], [8, 9, 12], [0, 5, 17], [4, 7, 10], [3, 9, 14], [3, 9, 17], [1, 2, 4], [11, 14, 16], [0, 2, 10], [1, 3, 6], [2, 8, 15], [5, 8, 16], [3, 4, 9], [2, 8, 14], [2, 5, 9], [1, 4, 5], [4, 6, 13], [9, 12, 17], [7, 12, 17], [3, 6, 16], [10, 13, 14], [5, 10, 17], [6, 11, 17], [5, 12, 15], [4, 12, 14], [2, 7, 17], [3, 12, 14], [0, 3, 15], [1, 4, 16], [6, 11, 15], [0, 4, 9], [0, 10, 14], [0, 10, 13], [4, 9, 16], [0, 6, 8], [0, 4, 14], [4, 9, 14], [2, 14, 16], [13, 14, 17], [0, 3, 9], [2, 8, 12], [7, 16, 17], [6, 14, 15], [3, 13, 15], [4, 10, 16], [9, 11, 15], [4, 7, 9], [4, 10, 11], [1, 2, 5], [6, 7, 8], [0, 3, 17], [3, 9, 15], [0, 13, 17], [3, 12, 15], [5, 6, 17], [4, 5, 13], [6, 10, 17], [1, 5, 10], [1, 2, 6], [6, 12, 14], [11, 13, 14], [1, 11, 17], [0, 6, 14], [0, 2, 8], [1, 2, 7], [1, 6, 8], [1, 3, 12], [3, 4, 11], [7, 12, 13], [12, 13, 14], [4, 7, 13], [2, 5, 14], [1, 2, 3], [7, 8, 10], [0, 4, 12], [0, 4, 6], [2, 3, 4], [8, 14, 16], [1, 3, 16], [1, 15, 16], [3, 15, 16], [0, 5, 13], [1, 9, 16], [6, 9, 15], [5, 9, 16], [1, 5, 14], [7, 13, 16], [1, 4, 15], [10, 11, 17], [1, 3, 4], [2, 12, 16], [0, 11, 17], [4, 9, 12], [5, 13, 17], [2, 6, 13], [4, 9, 15], [5, 7, 8], [8, 11, 13], [1, 9, 15], [9, 10, 16], [0, 11, 15], [0, 2, 6], [0, 1, 17], [1, 6, 17], [9, 11, 13], [2, 9, 12], [4, 7, 8], [1, 2, 12], [1, 11, 16], [4, 11, 13], [1, 11, 12], [4, 8, 15], [2, 3, 15], [2, 15, 16], [2, 10, 13], [0, 5, 12], [13, 15, 17], [3, 4, 13], [5, 16, 17], [9, 11, 12], [2, 10, 11], [3, 4, 14], [6, 10, 16], [8, 10, 15], [4, 5, 16], [4, 6, 10], [2, 3, 6], [8, 9, 14], [9, 12, 14], [5, 10, 11], [12, 14, 16], [0, 8, 9], [6, 16, 17], [4, 6, 9], [15, 16, 17], [7, 12, 14], [3, 5, 9], [1, 7, 12], [0, 8, 17], [0, 5, 11], [12, 13, 17], [6, 12, 16], [5, 10, 14], [1, 3, 11], [3, 9, 11], [7, 8, 16], [1, 4, 13], [1, 6, 9], [6, 9, 17], [8, 15, 16], [1, 3, 7], [0, 1, 5], [6, 12, 17], [3, 4, 17], [8, 11, 16], [11, 16, 17], [3, 8, 9], [7, 13, 17], [1, 6, 12], [1, 2, 13], [4, 13, 17], [0, 4, 11], [2, 6, 9], [2, 6, 11], [5, 6, 10], [7, 8, 9], [2, 3, 10], [0, 4, 13], [3, 6, 10], [6, 8, 14], [2, 7, 11], [4, 7, 12], [10, 13, 15], [1, 5, 12], [5, 14, 17], [2, 4, 7], [1, 4, 17], [4, 14, 15], [3, 8, 16], [12, 15, 16], [7, 8, 14], [2, 4, 12], [9, 14, 17], [3, 8, 14], [1, 4, 11], [0, 5, 9], [4, 8, 17], [1, 11, 15], [1, 4, 9], [0, 3, 16], [4, 8, 16], [0, 3, 6], [3, 8, 11], [4, 10, 17], [1, 7, 17], [8, 9, 10], [3, 4, 12], [2, 8, 9], [5, 11, 12], [1, 12, 16], [3, 9, 10], [2, 7, 8], [1, 3, 14], [4, 6, 16], [2, 5, 16], [1, 6, 11], [2, 13, 15], [5, 10, 16], [14, 15, 17], [0, 2, 15], [3, 6, 15], [2, 12, 13], [11, 14, 17], [2, 6, 7], [1, 2, 14], [1, 5, 6], [7, 10, 15], [7, 11, 12], [0, 6, 15], [3, 6, 7], [1, 11, 13]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    edge_order = P['edge_order']
    n = P['n']
    all_mask = (1 << n) - 1
    num_triples = len(edge_order)

    # Precompute masks for each triple
    triple_masks = []
    for t in edge_order:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)

    # Precompute vertex -> list of triple indices (sorted by index)
    vert_to_triples = [[] for _ in range(n)]
    for i, t in enumerate(edge_order):
        for v in t:
            vert_to_triples[v].append(i)

    # Precompute conflict bitmask for each triple: which other triples share a vertex
    # conflict_mask[i] is a bitset of indices j != i where triple i and j share a vertex
    # Since num_triples can be up to ~818, we use Python ints as arbitrary-precision bitsets
    conflict_mask = [0] * num_triples
    # For each pair of triples, if they share a vertex, mark conflict
    # Optimize: for each vertex, all triples containing it conflict with each other
    for v in range(n):
        triples_with_v = vert_to_triples[v]
        for a in range(len(triples_with_v)):
            ia = triples_with_v[a]
            for b in range(a + 1, len(triples_with_v)):
                ib = triples_with_v[b]
                conflict_mask[ia] |= (1 << ib)
                conflict_mask[ib] |= (1 << ia)

    def solve_with_prefix(k):
        """Find 6 disjoint triples from indices < k covering all vertices.
        Returns list of 6 indices or None."""
        if time.time() > deadline:
            return None

        # Quick feasibility check: every vertex must appear in at least one triple with index < k
        for v in range(n):
            has_candidate = False
            for idx in vert_to_triples[v]:
                if idx < k:
                    has_candidate = True
                    break
            if not has_candidate:
                return None

        # Build available triples list for this prefix
        avail = list(range(k))

        # For each vertex, precompute which available triples contain it
        vert_avail = [[] for _ in range(n)]
        for idx in avail:
            t = edge_order[idx]
            for v in t:
                vert_avail[v].append(idx)

        # Count nodes visited for timeout
        nodes_visited = [0]
        max_nodes = 5000000

        def backtrack(used_mask, chosen, path):
            if time.time() > deadline:
                return None
            nodes_visited[0] += 1
            if nodes_visited[0] > max_nodes:
                return None

            if chosen == 6:
                if used_mask == all_mask:
                    return path[:]
                return None

            # MRV: Find unused vertex with fewest valid candidates
            best_v = -1
            best_candidates = None
            min_count = float('inf')

            for v in range(n):
                if (used_mask >> v) & 1:
                    continue

                # Count valid candidates for v: disjoint with used_mask
                valid_cands = []
                for idx in vert_avail[v]:
                    if triple_masks[idx] & used_mask:
                        continue
                    valid_cands.append(idx)

                count = len(valid_cands)
                if count < min_count:
                    min_count = count
                    best_v = v
                    best_candidates = valid_cands
                    if count == 0:
                        break
                    if count == 1:
                        break

            if best_v == -1:
                return None
            if min_count == 0:
                return None

            # Pruning: check that every other unused vertex still has at least one candidate
            # after tentatively using best_v's candidates (we check without committing yet)
            # Actually, we'll do a lighter check: for each unused vertex != best_v,
            # verify at least one candidate doesn't conflict with used_mask
            for v2 in range(n):
                if v2 == best_v:
                    continue
                if (used_mask >> v2) & 1:
                    continue
                found = False
                for idx in vert_avail[v2]:
                    if triple_masks[idx] & used_mask:
                        continue
                    found = True
                    break
                if not found:
                    return None

            # Order candidates: prefer those that cover more "constrained" vertices
            # (vertices with fewer remaining candidates)
            def candidate_score(idx):
                t = edge_order[idx]
                score = 0
                for v in t:
                    if (used_mask >> v) & 1:
                        continue
                    # Count how many other candidates for v are still available
                    cnt = 0
                    for idx2 in vert_avail[v]:
                        if idx2 == idx:
                            continue
                        if triple_masks[idx2] & used_mask:
                            continue
                        cnt += 1
                    # Fewer remaining options = more constrained = higher priority
                    score += (min_count + 1) - min(cnt, min_count)
                return score

            # Sort candidates by score descending (most constraining first)
            best_candidates_sorted = sorted(best_candidates, key=candidate_score, reverse=True)

            # Try candidates in order
            for idx in best_candidates_sorted:
                path.append(idx)
                res = backtrack(used_mask | triple_masks[idx], chosen + 1, path)
                if res is not None:
                    return res
                path.pop()

            return None

        return backtrack(0, 0, [])

    # Binary search for the minimum prefix
    best_result = None
    lo = 6
    hi = num_triples

    # First verify a solution exists in the full list
    if time.time() < deadline - 10:
        test_res = solve_with_prefix(num_triples)
        if test_res is not None:
            best_result = test_res
            # Binary search for the minimum
            while lo <= hi and time.time() < deadline - 10:
                mid = (lo + hi) // 2
                res = solve_with_prefix(mid)
                if res is not None:
                    best_result = res
                    hi = mid - 1
                else:
                    lo = mid + 1
        else:
            return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
    else:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]

    # Refine: linear scan downward from the binary search result
    if best_result is not None and time.time() < deadline - 5:
        # After binary search, lo should be the answer (first k where solution exists)
        # Verify by scanning down from lo
        for k in range(lo, 5, -1):
            if time.time() > deadline - 2:
                break
            if k < lo:
                break
            res = solve_with_prefix(k)
            if res is not None:
                best_result = res
            else:
                break

    # Convert indices to triples
    if best_result is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]

    result_triples = []
    for idx in best_result:
        t = edge_order[idx]
        result_triples.append([int(t[0]), int(t[1]), int(t[2])])

    return result_triples

# EVOLVE-BLOCK-END
