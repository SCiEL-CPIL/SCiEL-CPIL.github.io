import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 14
    m = 35

    # Precompute all possible undirected edges on n vertices
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edge_select(values):
        """
        Deterministic greedy: at each step, pick the edge that minimizes
        the marginal max(distinct_sums, distinct_prods) increase.
        """
        sums = set()
        prods = set()
        selected = set()
        selected_list = []

        # Precompute edge contributions for all edges
        # For each edge (u,v), we know sum and product
        edge_data = []
        for (u, v) in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))

        for step in range(m):
            if time.time() > deadline - 5.0:
                break

            best_score = None
            best_edge = None
            best_new_sums = None
            best_new_prods = None
            best_marginal = None

            for (u, v, s, p) in edge_data:
                if (u, v) in selected:
                    continue
                # Marginal: how many new sums and prods does this edge add?
                new_s = 1 if s not in sums else 0
                new_p = 1 if p not in prods else 0
                # The resulting sizes
                ns = len(sums) + new_s
                np_ = len(prods) + new_p
                marginal = max(ns, np_)

                # Prefer lower marginal; tie-break: prefer edges that add
                # fewer NEW sums (additive collisions are harder to engineer
                # than multiplicative ones for small values)
                key = (marginal, new_s, new_p)

                if best_score is None or key < best_score:
                    best_score = key
                    best_edge = (u, v)
                    best_new_sums = sums | ({s} if new_s else set())
                    best_new_prods = prods | ({p} if new_p else set())
                    best_marginal = marginal

            if best_edge is None:
                break

            selected.add(best_edge)
            selected_list.append(best_edge)
            sums = best_new_sums
            prods = best_new_prods

        return selected_list, sums, prods

    # Candidate value sets: small structured sets encourage collisions
    # The key insight: with small values, sums and products overlap more
    value_candidates = [
        list(range(1, 15)),           # 1..14
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # same
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
    ]

    # Also try some with more spread but still structured
    value_candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20])
    value_candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20])
    value_candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20])

    best_score = float('inf')
    best_values = None
    best_edges = None

    for vals in value_candidates:
        if time.time() > deadline - 10.0:
            break

        # Ensure all values are distinct positive ints <= 10000
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > 10000 for v in vals):
            continue

        edges, sums, prods = greedy_edge_select(vals)

        if len(edges) < m:
            continue

        sc = max(len(sums), len(prods))

        if sc < best_score:
            best_score = sc
            best_values = list(vals)
            best_edges = list(edges)

    # If greedy didn't find enough edges or no valid config, fallback
    if best_edges is None or len(best_edges) < m:
        # Fallback: use first candidate with all edges
        vals = list(range(1, 15))
        edges = list(all_edges[:m])
        best_score = score_of(vals, edges)
        best_values = vals
        best_edges = edges

    # Ensure we have exactly m edges
    if len(best_edges) > m:
        best_edges = best_edges[:m]

    # Final validation
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]

    # Verify constraints
    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == m
    assert all(len(e) == 2 for e in final_edges)
    assert all(e[0] != e[1] for e in final_edges)
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    assert len(edge_set) == m

    return {
        'values': final_values,
        'edges': final_edges
    }
