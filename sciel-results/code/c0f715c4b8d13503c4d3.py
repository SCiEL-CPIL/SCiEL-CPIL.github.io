"""Solve the transversal problem: minimum set of vertices hitting every maximal clique (size>=2).

Strategy:
1. Generate each random graph with the same RNG procedure as the seed program.
2. Enumerate all maximal cliques (size >= 2) using Bron-Kerbosch with pivot.
3. Solve the hitting set problem via Integer Linear Programming using scipy.optimize.milp.
   If MILP is unavailable or fails, fall back to a greedy heuristic.
4. Use a time budget to ensure we finish well within the 150s wall-clock limit.
"""

import time
import sys
from random import Random

def _random_graph_edges(n, p, seed):
    """Generate edges of a random graph using the same procedure as the seed program."""
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        PX = P | X
        if not PX:
            return
        # Pick pivot with max degree in P
        best_pivot = None
        best_count = -1
        for u in PX:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_pivot = u
        # Iterate over P \ N(pivot)
        candidates = list(P - adj[best_pivot])
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(frozenset(), set(range(n)), set())
    return cliques

def _solve_hitting_set(n, cliques, time_limit=10.0):
    """Solve the minimum hitting set problem.
    
    Returns a list of vertex indices forming a minimum (or near-minimum) hitting set.
    """
    if not cliques:
        return []
    
    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        m = len(cliques)
        # Variables: x_v for v in range(n), binary, x_v = 1 if vertex v is in the hitting set
        # Constraints: for each clique C, sum_{v in C} x_v >= 1
        # Objective: minimize sum x_v
        
        # Build constraint matrix A (m x n)
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        # Constraints: A @ x >= 1  =>  -A @ x <= -1
        # LinearConstraint expects (lb, ub) for each row
        # We want A @ x >= 1, which is -A @ x <= -1
        # Or equivalently: A @ x >= 1 => lb=1, ub=inf for each row
        constraint = LinearConstraint(A, lb=1.0, ub=np.inf)
        integrality = np.ones(n)  # all variables binary
        bounds = Bounds(0, 1)
        
        c = np.ones(n)  # minimize sum x_v
        
        start_time = time.time()
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            # Extract vertices where x_v >= 0.5
            selected = [v for v in range(n) if result.x[v] >= 0.5]
            # Verify the solution
            if _verify_hitting_set(selected, cliques):
                return sorted(selected)
        
    except Exception:
        pass
    
    # Fallback: greedy heuristic
    return _greedy_hitting_set(n, cliques)

def _verify_hitting_set(selected, cliques):
    """Verify that selected vertices hit every clique."""
    selected_set = set(selected)
    for clique in cliques:
        if not (clique & selected_set):
            return False
    return True

def _greedy_hitting_set(n, cliques):
    """Greedy heuristic for minimum hitting set."""
    if not cliques:
        return []
    
    remaining_cliques = [set(c) for c in cliques]
    selected = set()
    
    # Sort cliques by size (smaller first) for better greedy performance
    remaining_cliques.sort(key=len)
    
    while remaining_cliques:
        # Find the vertex that appears in the most remaining cliques
        # For efficiency, count occurrences
        counts = {}
        for clique in remaining_cliques:
            for v in clique:
                if v not in selected:
                    counts[v] = counts.get(v, 0) + 1
        
        if not counts:
            break
        
        # Pick vertex with max count
        best_v = max(counts, key=counts.get)
        selected.add(best_v)
        
        # Remove all cliques that contain best_v
        remaining_cliques = [c for c in remaining_cliques if best_v not in c]
    
    return sorted(selected)

def _local_search_improve(selected, cliques, n, max_iters=100):
    """Try to improve the hitting set via local search (remove redundant vertices, swap)."""
    selected_set = set(selected)
    
    # First, remove any redundant vertices
    changed = True
    while changed:
        changed = False
        for v in list(selected_set):
            # Check if removing v still hits all cliques
            test_set = selected_set - {v}
            if _verify_hitting_set(sorted(test_set), cliques):
                selected_set = test_set
                changed = True
    
    # Try swaps: replace one vertex with another
    for _ in range(max_iters):
        improved = False
        for v in list(selected_set):
            test_set = selected_set - {v}
            # Find cliques not hit by test_set
            uncovered = [c for c in cliques if not (c & test_set)]
            if not uncovered:
                # v was redundant, remove it
                selected_set = test_set
                improved = True
                break
            # Try adding a single vertex w to hit all uncovered cliques
            # w must be in the intersection of all uncovered cliques
            if uncovered:
                common = set(uncovered[0])
                for c in uncovered[1:]:
                    common &= c
                for w in common:
                    if w not in selected_set:
                        new_set = test_set | {w}
                        if len(new_set) < len(selected_set):
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return sorted(selected_set)

def construct():
    """Main entry point. Returns three vertex lists, one per instance."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        # Check time
        if time.time() > deadline:
            # Fallback: return all vertices that are in any edge (guaranteed to hit all cliques of size>=2)
            edges = _random_graph_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            out.append([])
            continue
        
        # Solve hitting set
        selected = _solve_hitting_set(n, cliques, time_limit=10.0)
        
        # Local search improvement
        if time.time() < deadline - 5.0:
            selected = _local_search_improve(selected, cliques, n, max_iters=50)
        
        # Final verification
        if not _verify_hitting_set(selected, cliques):
            # Fallback to all touched vertices
            touched = {x for e in edges for x in e}
            selected = sorted(touched)
        
        out.append(sorted(selected))
    
    return out
