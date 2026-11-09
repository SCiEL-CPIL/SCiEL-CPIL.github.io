"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # safety margin under 120s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch(adj, R, P, X, maximal_cliques):
    """
    Bron-Kerbosch algorithm with pivot selection to find all maximal cliques.
    adj: list of sets of neighbors
    R: current clique
    P: candidates
    X: excluded candidates
    """
    if not P and not X:
        maximal_cliques.append(frozenset(R))
        return
    
    # Pivot selection: choose u from P U X with maximum degree in P
    # This minimizes the number of recursive calls
    pivot_set = P | X
    if pivot_set:
        # Find pivot u in pivot_set that maximizes |P \ N(u)|
        # Actually, standard BK uses u in P U X to minimize recursion.
        # We want u such that P \ N(u) is small? No, we iterate over P \ N(u).
        # So we want u with large N(u) intersection P?
        # Let's just pick the one with max degree in P.
        best_u = None
        max_deg = -1
        for u in pivot_set:
            deg = len(P & adj[u])
            if deg > max_deg:
                max_deg = deg
                best_u = u
        # We iterate over P \ N(best_u)
        candidates = P - adj[best_u]
    else:
        candidates = P
        
    for v in candidates:
        # Recursive call
        _bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], maximal_cliques)
        # Update P and X
        P.remove(v)
        X.add(v)

def _find_maximal_cliques(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    maximal_cliques = []
    _bron_kerbosch(adj, set(), set(range(n)), set(), maximal_cliques)
    return maximal_cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Map edges to indices for ILP
    edge_list = list(edges)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    
    # Find all maximal cliques
    maximal_cliques = _find_maximal_cliques(n, edges)
    
    # Filter cliques: must have at least 2 vertices
    valid_cliques = [c for c in maximal_cliques if len(c) >= 2]
    
    # If no valid cliques (should not happen if edges exist), fallback
    if not valid_cliques:
        return [[u, v] for u, v in edges]
        
    # Sort cliques by size descending to help ILP or just for consistent ordering
    valid_cliques.sort(key=lambda x: -len(x))
    k = len(valid_cliques)
    
    # Precompute for each clique, which edges it covers
    # clique_edges[ci] = set of edge indices covered by clique ci
    clique_edges = []
    for c in valid_cliques:
        c_list = list(c)
        covered = set()
        for u, v in combinations(c_list, 2):
            e = (min(u, v), max(u, v))
            if e in edge_to_idx:
                covered.add(edge_to_idx[e])
        clique_edges.append(covered)
        
    # ILP Formulation:
    # Minimize sum(x_i) for i in 0..k-1
    # Subject to:
    #   sum(x_i for i where e in clique_edges[i]) == 1 for all e in 0..m-1
    #   x_i in {0, 1}
    
    # Since scipy.optimize.milp might be slow or unavailable in some strict envs, 
    # but the prompt says SciPy is available. 
    # However, m can be up to ~2000 for n=64, p=0.55. k (max cliques) can be large.
    # For n=32, k is smaller.
    # If k is too large, ILP might time out.
    # Fallback: Greedy Set Packing (Largest First) if ILP fails or takes too long.
    
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # Objective: c = [1, 1, ..., 1]
    c = np.ones(k)
    
    # Constraints: A x = 1
    # A is m x k
    # A[e, i] = 1 if edge e is in clique i, else 0
    # This matrix is sparse. scipy.optimize.milp accepts sparse or dense.
    # For m~2000, k~1000, dense is 2M floats, which is fine.
    
    # Build A
    A = np.zeros((m, k))
    for i, covered in enumerate(clique_edges):
        for e_idx in covered:
            A[e_idx, i] = 1.0
            
    # Constraint: A x = 1
    # Lower bound 1, Upper bound 1
    constraint = LinearConstraint(A, 1, 1)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all variables are integer
    integrality = np.ones(k)
    
    start_time = time.time()
    
    try:
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.success:
            # Extract solution
            x = res.x
            # x should be 0 or 1 (approximately)
            selected_cliques = []
            for i in range(k):
                if x[i] > 0.5:
                    selected_cliques.append(list(valid_cliques[i]))
            
            # Verify validity just in case (numerical errors)
            # Check if every edge is covered exactly once
            covered_count = [0] * m
            for c_list in selected_cliques:
                for u, v in combinations(c_list, 2):
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        covered_count[edge_to_idx[e]] += 1
            
            if all(cnt == 1 for cnt in covered_count):
                return selected_cliques
            else:
                # If invalid, fallback
                pass
        else:
            # ILP failed, fallback
            pass
            
    except Exception:
        # Fallback
        pass
        
    # Fallback: Greedy Largest-First Set Packing
    # This is a heuristic that works well in practice
    uncovered_edges = set(range(m))
    partition = []
    
    # Sort cliques by size descending
    # We need to be careful: picking a large clique might block multiple smaller ones,
    # but usually large cliques are better for minimizing count.
    # Greedy: pick the clique that covers the most uncovered edges.
    
    # To make it faster, we can just pick the largest clique that covers at least 1 uncovered edge?
    # Or better: iterate through cliques sorted by size, pick if it covers >0 uncovered, remove those edges.
    # This is "Greedy Maximum Clique Cover" but edge-disjoint.
    
    # Let's do a slightly smarter greedy:
    # At each step, find the clique that covers the most uncovered edges.
    # This is O(k * m) per step, which is too slow if k, m are large.
    # But we have time limit.
    
    # Simpler fallback: Just take all edges as cliques if ILP fails? No, that's score 1.0.
    # Let's use the greedy expansion from the current program as a fallback?
    # Or just a simple greedy:
    
    # Let's try: Sort cliques by size descending.
    # Iterate. If clique covers any uncovered edge, take it, mark its edges covered.
    # This is not optimal but better than edges.
    
    cliques_sorted = sorted(valid_cliques, key=lambda x: -len(x))
    for c in cliques_sorted:
        if not uncovered_edges:
            break
        c_list = list(c)
        covered = []
        for u, v in combinations(c_list, 2):
            e = (min(u, v), max(u, v))
            if e in edge_to_idx:
                idx = edge_to_idx[e]
                if idx in uncovered_edges:
                    covered.append(idx)
        if covered:
            partition.append(c_list)
            for idx in covered:
                uncovered_edges.remove(idx)
                
    # Handle remaining edges
    for idx in uncovered_edges:
        u, v = edge_list[idx]
        partition.append([u, v])
        
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if not _deadline_ok():
            # Emergency fallback: return empty or minimal valid?
            # Must return valid partition.
            # If no time, we can't compute edges.
            # But we should have computed edges quickly.
            # Let's assume we have time for at least the first instance.
            pass
        
        results.append(_solve_instance(n, p, seed))
        
    return results
# EVOLVE-BLOCK-END
