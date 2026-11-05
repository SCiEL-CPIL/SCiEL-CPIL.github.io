"""Rewritten program: spectral heuristic + local search for Q3-free graph on 16 vertices."""
import time
import random
import numpy as np

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds to stay under the limit

def _now():
    return time.time()

def _edges_from_adj(adj):
    """Return list of [u, v] edges from a 16x16 boolean adjacency matrix (u < v)."""
    edges = []
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def _has_q3(adj):
    """
    Check whether the graph with adjacency matrix adj contains a subgraph isomorphic to Q3.
    Q3 has 8 vertices and 12 edges. We check all 8-vertex subsets for the Q3 structure.
    For efficiency, we use the fact that Q3 is bipartite with parts of size 4, and
    each vertex has degree 3 within the Q3 subgraph.
    """
    n = adj.shape[0]
    # Precompute degrees
    deg = adj.sum(axis=1).astype(int)
    
    # For a Q3 subgraph, all 8 vertices must have degree exactly 3 within the subgraph.
    # We enumerate 4-sets for each bipartition side.
    from itertools import combinations
    
    # Get all 4-subsets of vertices
    all_4sets = list(combinations(range(n), 4))
    
    # For each pair of disjoint 4-sets (A, B), check if they form a Q3
    # Q3: bipartite, each vertex in A connected to exactly 3 in B, each in B to exactly 3 in A
    # Total edges between A and B must be 12 (each of 4 vertices in A has 3 neighbors in B)
    
    # Build index: map frozenset to index for fast lookup
    # Actually, let's just iterate over pairs of 4-sets that are disjoint
    
    for i, A in enumerate(all_4sets):
        A_set = set(A)
        for j in range(i + 1, len(all_4sets)):
            B = all_4sets[j]
            B_set = set(B)
            # Must be disjoint
            if A_set & B_set:
                continue
            # Check bipartite structure: each vertex in A has exactly 3 neighbors in B
            valid = True
            for u in A:
                cnt = sum(adj[u, v] for v in B)
                if cnt != 3:
                    valid = False
                    break
            if not valid:
                continue
            for v in B:
                cnt = sum(adj[u, v] for u in A)
                if cnt != 3:
                    valid = False
                    break
            if valid:
                return True
    return False

def _has_q3_fast(adj, bitmask_adj=None):
    """
    Faster Q3 detection using bitmask representation.
    adj: 16x16 numpy boolean matrix
    bitmask_adj: list of 16 integers, bitmask_adj[i] has bit j set if edge (i,j) exists
    """
    if bitmask_adj is None:
        bitmask_adj = [0] * 16
        for i in range(16):
            for j in range(16):
                if adj[i, j]:
                    bitmask_adj[i] |= (1 << j)
    
    from itertools import combinations
    
    all_4sets = list(combinations(range(16), 4))
    
    for i, A in enumerate(all_4sets):
        A_set = set(A)
        for j in range(i + 1, len(all_4sets)):
            B = all_4sets[j]
            B_set = set(B)
            if A_set & B_set:
                continue
            # Check: each u in A has exactly 3 neighbors in B
            valid = True
            for u in A:
                mask_B = 0
                for v in B:
                    mask_B |= (1 << v)
                # Count bits in (bitmask_adj[u] & mask_B)
                cnt = bin(bitmask_adj[u] & mask_B).count('1')
                if cnt != 3:
                    valid = False
                    break
            if not valid:
                continue
            for v in B:
                mask_A = 0
                for u in A:
                    mask_A |= (1 << u)
                cnt = bin(bitmask_adj[v] & mask_A).count('1')
                if cnt != 3:
                    valid = False
                    break
            if valid:
                return True
    return False

def _build_bitmask(adj):
    bm = [0] * 16
    for i in range(16):
        for j in range(16):
            if adj[i, j]:
                bm[i] |= (1 << j)
    return bm

def _spectral_remove_one(adj):
    """
    Find the edge whose removal most reduces trace(A^4) and remove it.
    Returns (new_adj, edge_removed) or (adj, None) if no edges.
    """
    n = adj.shape[0]
    A = adj.astype(float)
    
    # Compute A^2 and A^4
    A2 = A @ A
    A4 = A2 @ A2
    trace_A4 = np.trace(A4)
    
    best_edge = None
    best_reduction = 0.0
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                # Remove edge (u,v)
                # New A = A - e_u - e_v (where e_u has 1 at (u,v) and (v,u))
                # trace(A^4) change: we can compute incrementally
                # A' = A - (e_u e_v^T + e_v e_u^T)
                # This is expensive to compute fully; instead, note that
                # trace(A^4) = sum_{i,j} (A^2)_{i,j}^2 ... no, trace(A^4) = sum_i (A^4)_{i,i}
                # Let's just compute it directly for each candidate edge.
                # For 16 vertices, this is 120 edges * O(n^3) = manageable
                
                A_mod = A.copy()
                A_mod[u, v] = 0
                A_mod[v, u] = 0
                A2_mod = A_mod @ A_mod
                A4_mod = A2_mod @ A2_mod
                trace_mod = np.trace(A4_mod)
                reduction = trace_A4 - trace_mod
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_edge = (u, v)
    
    if best_edge is None:
        return adj, None
    
    new_adj = adj.copy()
    u, v = best_edge
    new_adj[u, v] = False
    new_adj[v, u] = False
    return new_adj, best_edge

def _local_search(adj, time_limit):
    """
    Local search: try removing edges to eliminate Q3, then try re-adding edges.
    """
    n = adj.shape[0]
    bm = _build_bitmask(adj)
    
    # First, ensure no Q3 by removing edges
    while _has_q3_fast(adj, bm):
        if _now() > time_limit:
            break
        # Find a Q3 and remove one of its edges
        # For now, just remove a random edge that's part of some Q3
        # Simple approach: remove random edge and check
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append((u, v))
        random.shuffle(edges)
        found = False
        for (u, v) in edges:
            adj[u, v] = False
            adj[v, u] = False
            bm[u] &= ~(1 << v)
            bm[v] &= ~(1 << u)
            if not _has_q3_fast(adj, bm):
                found = True
                break
            # Restore
            adj[u, v] = True
            adj[v, u] = True
            bm[u] |= (1 << v)
            bm[v] |= (1 << u)
        if not found:
            # Remove a random edge
            if edges:
                u, v = edges[0]
                adj[u, v] = False
                adj[v, u] = False
                bm[u] &= ~(1 << v)
                bm[v] &= ~(1 << u)
    
    # Now try to add edges back
    improved = True
    while improved and _now() < time_limit:
        improved = False
        # Try adding each missing edge
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    adj[u, v] = True
                    adj[v, u] = True
                    bm[u] |= (1 << v)
                    bm[v] |= (1 << u)
                    if not _has_q3_fast(adj, bm):
                        improved = True
                    else:
                        # Remove it
                        adj[u, v] = False
                        adj[v, u] = False
                        bm[u] &= ~(1 << v)
                        bm[v] &= ~(1 << u)
                    if _now() > time_limit:
                        return adj
    return adj

def _spectral_heuristic(time_limit):
    """
    Start from K16, iteratively remove edges to reduce trace(A^4) until no Q3.
    """
    n = 16
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    
    while _now() < time_limit:
        bm = _build_bitmask(adj)
        if not _has_q3_fast(adj, bm):
            break
        adj, edge = _spectral_remove_one(adj)
        if edge is None:
            break
    
    # Local search to add back edges
    adj = _local_search(adj, time_limit)
    return adj

def _random_greedy(time_limit):
    """
    Randomized greedy: start from empty graph, add edges in random order if no Q3 created.
    """
    n = 16
    adj = np.zeros((n, n), dtype=bool)
    bm = [0] * 16
    
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(edges)
    
    for (u, v) in edges:
        if _now() > time_limit:
            break
        adj[u, v] = True
        adj[v, u] = True
        bm[u] |= (1 << v)
        bm[v] |= (1 << u)
        if _has_q3_fast(adj, bm):
            adj[u, v] = False
            adj[v, u] = False
            bm[u] &= ~(1 << v)
            bm[v] &= ~(1 << u)
    
    return adj

def _k13_plus_cycle(time_limit):
    """
    Start from K3,13 and add cycle edges, then local search.
    """
    n = 16
    adj = np.zeros((n, n), dtype=bool)
    # K3,13: vertices 0,1,2 connected to 3..15
    for a in range(3):
        for b in range(3, 16):
            adj[a, b] = True
            adj[b, a] = True
    # Add cycle among 3..15
    for i in range(3, 16):
        j = 3 + ((i - 3 + 1) % 13)
        adj[i, j] = True
        adj[j, i] = True
    
    return _local_search(adj, time_limit)

def _k88_bipartite(time_limit):
    """
    Start from K8,8 bipartite graph, then local search.
    """
    n = 16
    adj = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = True
            adj[v, u] = True
    
    return _local_search(adj, time_limit)

def construct():
    start_time = _now()
    time_limit = start_time + _DEADLINE - _SAFETY
    
    best_adj = None
    best_edges = -1
    
    # Strategy 1: Spectral heuristic
    if _now() < time_limit:
        adj = _spectral_heuristic(time_limit)
        edges = _edges_from_adj(adj)
        if not _has_q3_fast(adj, _build_bitmask(adj)):
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = adj
    
    # Strategy 2: Random greedy with multiple restarts
    remaining = time_limit - _now()
    num_restarts = max(1, int(remaining / 5))  # ~5s per restart
    for r in range(num_restarts):
        if _now() >= time_limit:
            break
        adj = _random_greedy(time_limit)
        edges = _edges_from_adj(adj)
        if not _has_q3_fast(adj, _build_bitmask(adj)):
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = adj
    
    # Strategy 3: K3,13 + cycle
    if _now() < time_limit:
        adj = _k13_plus_cycle(time_limit)
        edges = _edges_from_adj(adj)
        if not _has_q3_fast(adj, _build_bitmask(adj)):
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = adj
    
    # Strategy 4: K8,8
    if _now() < time_limit:
        adj = _k88_bipartite(time_limit)
        edges = _edges_from_adj(adj)
        if not _has_q3_fast(adj, _build_bitmask(adj)):
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = adj
    
    # Fallback: K3,13
    if best_adj is None:
        best_adj = np.zeros((16, 16), dtype=bool)
        for a in range(3):
            for b in range(3, 16):
                best_adj[a, b] = True
                best_adj[b, a] = True
    
    return _edges_from_adj(best_adj)
