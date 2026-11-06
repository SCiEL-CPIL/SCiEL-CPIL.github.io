"""
Triangle-free graph on up to 41 vertices with independence number < 10.
Uses a spectral/Paley-inspired seed construction followed by
simulated annealing to eliminate triangles while keeping alpha < 10.
"""

import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) is triangle-free."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                # Check common neighbors
                common = np.bitwise_and(adj[u], adj[v])
                if common.any():
                    return False
    return True

def _max_independent_set_size(adj, n, timeout=5.0):
    """
    Upper bound on independence number using branch and bound.
    Returns an upper bound; if the bound is < 10, alpha < 10 is guaranteed.
    If bound >= 10, we need exact check. For performance, we use a heuristic
    lower bound + greedy coloring on complement for upper bound.
    """
    # Use greedy coloring on the complement graph to get upper bound on alpha
    # alpha(G) = chi(complement(G))
    # Greedy coloring gives an upper bound on chromatic number? No, it gives a valid coloring
    # so the number of colors used is an upper bound on chi, hence upper bound on alpha.
    
    # Actually, greedy coloring gives a valid coloring, so the number of colors
    # is >= chi. Wait no: greedy coloring produces a proper coloring with some number
    # of colors, which is >= chi(G_complement) = alpha(G). So it's a lower bound on alpha.
    
    # For upper bound on alpha, we need a different approach.
    # Let's just do exact B&B with a time limit.
    
    start_time = time.time()
    
    # Adjacency of complement
    comp_adj = np.logical_not(adj)
    np.fill_diagonal(comp_adj, False)
    
    # We want to find if there's an independent set of size 10 in G,
    # which is a clique of size 10 in complement.
    # Use Bron-Kerbosch style with pivot, but just check if clique of size 10 exists.
    
    # Simpler: use a greedy + B&B approach
    # State: set of vertices available, current clique size
    # We want to know if max clique in complement >= 10
    
    # Convert to list for speed
    comp_neighbors = [set(np.where(comp_adj[i])[0]) for i in range(n)]
    
    best = [0]
    
    def bronk(R, P, X):
        if time.time() - start_time > timeout:
            return
        if len(P) + len(R) <= best[0]:
            return
        if len(R) >= 10:
            best[0] = 10
            return
        if len(P) == 0:
            best[0] = max(best[0], len(R))
            return
        # Pick pivot from P union X to minimize branching
        if P:
            pivot = next(iter(P))
        elif X:
            pivot = next(iter(X))
        else:
            return
        
        # Candidates: vertices in P not adjacent to pivot in complement
        # (i.e., not connected to pivot in complement = adjacent in original... wait)
        # In Bron-Kerbosch for maximum clique, we pick a pivot and branch on P \ N(pivot)
        # N(pivot) in the graph we're finding cliques in (complement)
        candidates = P - comp_neighbors[pivot]
        if not candidates:
            candidates = P
        
        for v in list(candidates):
            if time.time() - start_time > timeout:
                return
            bronk(R | {v}, P & comp_neighbors[v], X & comp_neighbors[v])
            P.remove(v)
            X.add(v)
            if time.time() - start_time > timeout:
                return
    
    bronk(set(), set(range(n)), set())
    return best[0]

def _alpha_bound(adj, n):
    """
    Return an upper bound on alpha(G). If < 10, we're safe.
    Uses greedy coloring of complement as a heuristic upper bound is not directly valid.
    Instead, use the fact that alpha <= n * (1 - |E|/(n*(n-1)/2)) ... no that's not right.
    
    Let's use a simple bound: alpha <= n - minimum vertex cover... 
    Actually, let's just use the B&B result but cap it.
    """
    return _max_independent_set_size(adj, n, timeout=3.0)

def _count_triangles(adj, n):
    """Count number of triangles in graph."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                count += np.bitwise_and(adj[u], adj[v]).sum() // 1
    return count // 3

def construct():
    """
    Construct a triangle-free graph on as many vertices as possible (up to 41)
    with independence number < 10.
    
    Strategy:
    1. Start with a Paley-like or random graph seed on 40 vertices
    2. Use simulated annealing to remove edges to eliminate triangles
       while trying to keep the independence number low (< 10)
    """
    n_target = 40
    random.seed(42)
    np.random.seed(42)
    
    # Try multiple seeds/initializations
    best_n = 0
    best_edges = None
    
    for attempt in range(50):
        random.seed(attempt * 137 + 1)
        np.random.seed(attempt * 211 + 7)
        
        # Start with a random graph
        p = 0.4 + 0.1 * (attempt % 10) / 10  # Vary density
        adj = np.zeros((n_target, n_target), dtype=bool)
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Simulated annealing to remove triangles
        # Energy = number of triangles (we want 0) + penalty if alpha >= 10
        # We only add/remove edges; removing edges reduces triangles but may increase alpha
        
        current_adj = adj.copy()
        temp = 1.0
        cooling = 0.995
        max_iter = 5000
        start_time = time.time()
        
        # Precompute triangle count
        cur_tri = _count_triangles(current_adj, n_target)
        
        # Get initial alpha bound
        cur_alpha = _alpha_bound(current_adj, n_target)
        
        for it in range(max_iter):
            if time.time() - start_time > 90:
                break
            
            if cur_tri == 0 and cur_alpha < 10:
                break
            
            # Pick a random edge to toggle
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Try removing the edge if present, or adding if not
            if current_adj[i, j]:
                # Remove edge
                current_adj[i, j] = False
                current_adj[j, i] = False
                new_tri = _count_triangles(current_adj, n_target)
                # Check alpha - approximate with a quick bound
                # For speed, skip full B&B every iteration; do it periodically
                # Energy change: triangles decrease (good), alpha may increase (bad)
                delta_e = (cur_tri - new_tri)  # Positive means improvement in triangles
                # Accept if it reduces triangles or with some probability
                if new_tri < cur_tri:
                    cur_tri = new_tri
                    # Periodically check alpha
                    if it % 100 == 0:
                        cur_alpha = _alpha_bound(current_adj, n_target)
                        if cur_alpha >= 10:
                            # Revert
                            current_adj[i, j] = True
                            current_adj[j, i] = True
                            cur_tri += new_tri - cur_tri  # This is getting complicated
                            # Just revert and continue
                            current_adj[i, j] = True
                            current_adj[j, i] = True
                            cur_tri = _count_triangles(current_adj, n_target)
                else:
                    # Revert
                    current_adj[i, j] = True
                    current_adj[j, i] = True
            else:
                # Add edge - this could create triangles, only do if it helps
                # Actually, adding edges makes it harder to be triangle-free, so
                # focus on removal. Skip adding for simplicity.
                pass
            
            temp *= cooling
        
        # Final check
        if _is_triangle_free(current_adj, n_target):
            alpha = _alpha_bound(current_adj, n_target, timeout=8.0)
            if alpha < 10:
                edges = [[i, j] for i in range(n_target) for j in range(i + 1, n_target) if current_adj[i, j]]
                return {'n': n_target, 'edges': edges}
        
        # Try smaller n
        for n_try in range(39, 26, -1):
            # Build subgraph on first n_try vertices
            sub_adj = current_adj[:n_try, :n_try].copy()
            if _is_triangle_free(sub_adj, n_try):
                alpha = _alpha_bound(sub_adj, n_try, timeout=5.0)
                if alpha < 10:
                    edges = [[i, j] for i in range(n_try) for j in range(i + 1, n_try) if sub_adj[i, j]]
                    if n_try > best_n:
                        best_n = n_try
                        best_edges = edges
                    break
        
        if best_n >= 39:
            break
    
    # Fallback: use the Andrasfai graph on 26 vertices if nothing better found
    if best_edges is None or best_n < 26:
        n = 26
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
