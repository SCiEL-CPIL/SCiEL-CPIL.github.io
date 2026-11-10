"""Clique partition of two fixed dense random graphs via randomized greedy with restarts."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _greedy_partition(edges, n, rng):
    """
    Greedy clique partition: repeatedly extract a large clique from remaining edges.
    Uses a randomized vertex ordering to diversify solutions.
    """
    if not edges:
        return []

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    remaining = set(edge_set)
    cliques = []

    # Precompute adjacency for faster lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Randomize vertex order for diversity
    vertex_order = list(range(n))
    rng.shuffle(vertex_order)

    while remaining:
        if time.time() > _DEADLINE - 5:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                cliques.append([e[0], e[1]])
            break

        # Find vertex with max degree in remaining graph, preferring earlier in shuffled order
        best_v = -1
        best_deg = -1
        best_order = float('inf')
        for v in vertex_order:
            deg = len(adj[v] & {u for u, v2 in remaining for v2 in [u] if v == v2} |
                         {v2 for u, v2 in remaining if v == u} |
                         {u for u, v2 in remaining if v == v2})
            # Recompute degree properly
            deg = 0
            for u, w in remaining:
                if u == v:
                    deg += 1
                elif w == v:
                    deg += 1
            if deg > best_deg or (deg == best_deg and vertex_order.index(v) < best_order):
                best_deg = deg
                best_v = v
                best_order = vertex_order.index(v)
        if best_v == -1 or best_deg == 0:
            # Shouldn't happen if remaining is non-empty
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            continue

        # Build clique starting with best_v using greedy extension
        C = [best_v]
        # Candidates: neighbors of best_v in remaining graph
        candidates = set()
        for u, w in remaining:
            if u == best_v:
                candidates.add(w)
            elif w == best_v:
                candidates.add(u)

        # Greedily extend: at each step, pick candidate with max degree within candidates
        while candidates:
            # Find candidate with max degree within current candidates
            best_u = -1
            best_deg_u = -1
            for u in candidates:
                d = 0
                for w in candidates:
                    if u == w:
                        continue
                    if (min(u, w), max(u, w)) in remaining:
                        d += 1
                if d > best_deg_u:
                    best_deg_u = d
                    best_u = u

            if best_u == -1:
                break

            C.append(best_u)

            # Update candidates: keep only those connected to best_u in remaining
            new_candidates = set()
            for w in candidates:
                if w == best_u:
                    continue
                if (min(best_u, w), max(best_u, w)) in remaining:
                    new_candidates.add(w)
            candidates = new_candidates

        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = (min(u, w), max(u, w))
                    remaining.discard(e)
        else:
            # Fallback: take one edge
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))

    return cliques


def _solve_instance(edges, n, max_restarts=None):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    if max_restarts is None:
        # Estimate number of restarts based on remaining time
        remaining_time = _DEADLINE - 5 - time.time()
        if remaining_time < 5:
            max_restarts = 1
        elif n <= 32:
            max_restarts = min(50, int(remaining_time * 5))
        else:
            max_restarts = min(30, int(remaining_time * 3))

    best_cliques = None
    best_count = float('inf')

    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 5:
            break

        rng = _random.Random(1000 + restart)
        cliques = _greedy_partition(edges, n, rng)

        # Validate: every edge must be in exactly one clique, each clique must be a clique
        edge_count = 0
        for c in cliques:
            k = len(c)
            if k < 2:
                break
            edge_count += k * (k - 1) // 2
        else:
            # Check that total edges match
            if edge_count == len(edges):
                # Verify each clique is actually a clique in the original graph
                valid = True
                edge_set = set((min(u, v), max(u, v)) for u, v in edges)
                seen_edges = set()
                for c in cliques:
                    for i in range(len(c)):
                        for j in range(i + 1, len(c)):
                            e = (min(c[i], c[j]), max(c[i], c[j]))
                            if e not in edge_set or e in seen_edges:
                                valid = False
                                break
                            seen_edges.add(e)
                    if not valid:
                        break
                if valid and len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques

    if best_cliques is None:
        # Fallback: single restart without validation
        rng = _random.Random(0)
        best_cliques = _greedy_partition(edges, n, rng)

    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    res1 = _solve_instance(edges1, 32)
    res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
