"""Clique partition of two fixed dense random graphs via optimized bitmask greedy with restarts."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _greedy_partition_bitmask(edges, n, rng):
    """
    Greedy clique partition using bitmasks for speed.
    """
    if not edges:
        return []

    # Build initial adjacency bitmasks
    rem_adj = [0] * n
    for u, v in edges:
        rem_adj[u] |= (1 << v)
        rem_adj[v] |= (1 << u)
    
    # Track remaining edges count to stop early
    remaining_edges = len(edges)
    if remaining_edges == 0:
        return []

    cliques = []
    
    # To speed up finding max degree vertex, we can just iterate.
    # n is small (32, 64), so O(n) per iteration is fine.
    # Total iterations <= n^2/2 / 2 = ~500 for n=64.
    # Each iteration involves clique finding.
    
    while remaining_edges > 0:
        if time.time() > _DEADLINE - 2:
            # Fallback: just take one edge at a time to ensure termination and validity
            # This is a safety net, shouldn't happen if algorithm is fast
            # Find any edge
            for u in range(n):
                mask = rem_adj[u]
                if mask:
                    v = (mask & -mask).bit_length() - 1
                    cliques.append([u, v])
                    # Remove edge
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                    remaining_edges -= 1
                    if remaining_edges == 0:
                        break
            continue

        # Find vertex with max degree in remaining graph
        best_v = -1
        max_deg = -1
        # Randomize tie-breaking to diversify
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        for v in vertices:
            deg = rem_adj[v].bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
                if deg == 0:
                    # If all degrees are 0, we are done? No, remaining_edges > 0 implies some edge exists.
                    # But if max_deg is 0, then no edges. Contradiction.
                    pass

        if best_v == -1 or max_deg == 0:
            # Should not happen if remaining_edges > 0
            break

        # Build clique starting with best_v
        C = [best_v]
        # Candidates are neighbors of best_v
        candidates = rem_adj[best_v]
        
        # Greedy extension
        while candidates:
            # Find candidate with max degree within candidates
            # Degree of u within candidates = popcount(rem_adj[u] & candidates)
            best_u = -1
            best_deg_u = -1
            
            # Iterate through set bits in candidates
            temp_cand = candidates
            while temp_cand:
                lsb = temp_cand & -temp_cand
                u = lsb.bit_length() - 1
                temp_cand ^= lsb
                
                deg_u = (rem_adj[u] & candidates).bit_count()
                if deg_u > best_deg_u:
                    best_deg_u = deg_u
                    best_u = u
            
            if best_u == -1:
                break
                
            C.append(best_u)
            
            # Update candidates: must be connected to best_u
            candidates &= rem_adj[best_u]
            # Also remove best_u from candidates (it's in C now)
            candidates &= ~(1 << best_u)

        if len(C) >= 2:
            cliques.append(C)
            
            # Remove all edges within C from rem_adj
            # For each pair (u, v) in C, remove edge
            for i in range(len(C)):
                u = C[i]
                for j in range(i + 1, len(C)):
                    v = C[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                    remaining_edges -= 1
        else:
            # Fallback: single edge
            # This shouldn't happen if max_deg > 0, but just in case
            u = best_v
            mask = rem_adj[u]
            if mask:
                v = (mask & -mask).bit_length() - 1
                cliques.append([u, v])
                rem_adj[u] &= ~(1 << v)
                rem_adj[v] &= ~(1 << u)
                remaining_edges -= 1

    return cliques


def _validate_partition(cliques, edges, n):
    """
    Validate that cliques form a valid edge partition.
    Returns True if valid, False otherwise.
    """
    if not cliques:
        return len(edges) == 0

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    seen_edges = set()
    
    for c in cliques:
        if len(c) < 2:
            return False
        # Check if c is a clique in the original graph
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                u, v = c[i], c[j]
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    return False
                if e in seen_edges:
                    return False
                seen_edges.add(e)
                
    # Check if all edges are covered
    return seen_edges == edge_set


def _solve_instance(edges, n, max_restarts=None):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    if max_restarts is None:
        # Estimate number of restarts based on remaining time
        remaining_time = _DEADLINE - 5 - time.time()
        if remaining_time < 2:
            max_restarts = 1
        elif n <= 32:
            # Bitmask is fast, allow many restarts
            max_restarts = min(200, int(remaining_time * 50))
        else:
            max_restarts = min(100, int(remaining_time * 20))

    best_cliques = None
    best_count = float('inf')

    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 2:
            break

        rng = _random.Random(1000 + restart)
        cliques = _greedy_partition_bitmask(edges, n, rng)

        # Validate
        if _validate_partition(cliques, edges, n):
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

    if best_cliques is None:
        # Fallback: single restart without strict validation check in loop (but still valid structure)
        rng = _random.Random(0)
        best_cliques = _greedy_partition_bitmask(edges, n, rng)
        # If this fails validation, we have a problem, but it's a valid heuristic output
        # The verifier will reject it, but we must return something.
        # Usually the first run is valid.

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
