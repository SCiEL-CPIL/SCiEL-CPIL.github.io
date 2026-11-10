"""Spectral/Paley-inspired construction for K2,3-free graph on 20 vertices."""
import itertools
import random
import numpy as np

def _is_k23_free(adj):
    """Check if graph (adjacency matrix) is K2,3-free."""
    n = adj.shape[0]
    for i, j in itertools.combinations(range(n), 2):
        if adj[i, j]:
            continue
        common = adj[i] & adj[j]
        if common.sum() >= 3:
            return False
    return True

def _count_edges(adj):
    return int(adj.sum() // 2)

def _paley_graph(n):
    """Construct a Paley-type graph on n vertices using quadratic residues."""
    # For n=20, we can't use a standard Paley graph directly since 20 is not a prime power ≡ 1 mod 4.
    # Instead, we use a heuristic: map vertices to elements of Z_19 and use QR, then add vertex 20 carefully.
    # Actually, let's try a different approach: use a subset of vertices from a larger structure.
    
    # Let's try: use Z_19 with quadratic residues, giving 19 vertices. Then add a 20th vertex.
    p = 19
    # Quadratic residues mod 19
    qr = set((x * x) % p for x in range(p))
    qr.discard(0)
    
    # Build graph on 19 vertices
    adj = np.zeros((19, 19), dtype=bool)
    for i in range(19):
        for d in qr:
            j = (i + d) % 19
            if i < j:
                adj[i, j] = True
                adj[j, i] = True
    
    # Now add vertex 19 (index 19)
    # Connect it to some vertices to maximize edges while maintaining K2,3-free
    # Start by connecting to all, then remove edges that create K2,3
    adj20 = np.zeros((20, 20), dtype=bool)
    adj20[:19, :19] = adj
    
    # Try all subsets of connections for vertex 19, pick the best that is K2,3-free
    best_adj = None
    best_edges = 0
    
    # Heuristic: try connecting to vertices with certain properties
    # First, try all individual additions
    for subset_size in range(20, -1, -1):
        found = False
        # For efficiency, only try a few random subsets if full search is too slow
        if subset_size > 15:
            # Try random subsets
            for _ in range(100):
                subset = set(random.sample(range(19), subset_size))
                adj20[:19, :19] = adj
                for v in subset:
                    adj20[19, v] = True
                    adj20[v, 19] = True
                if _is_k23_free(adj20):
                    e = _count_edges(adj20)
                    if e > best_edges:
                        best_edges = e
                        best_adj = adj20.copy()
                    found = True
        else:
            # Try all combinations for smaller sizes
            for subset in itertools.combinations(range(19), subset_size):
                adj20[:19, :19] = adj
                for v in subset:
                    adj20[19, v] = True
                    adj20[v, 19] = True
                if _is_k23_free(adj20):
                    e = _count_edges(adj20)
                    if e > best_edges:
                        best_edges = e
                        best_adj = adj20.copy()
                    found = True
        if found:
            break
    
    if best_adj is not None:
        return best_adj
    return adj20

def _optimize(adj, max_iters=5000):
    """Local optimization via edge swaps and additions."""
    n = adj.shape[0]
    best_adj = adj.copy()
    best_edges = _count_edges(best_adj)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding edges
        for i, j in itertools.combinations(range(n), 2):
            if not best_adj[i, j]:
                best_adj[i, j] = True
                best_adj[j, i] = True
                if _is_k23_free(best_adj):
                    e = _count_edges(best_adj)
                    if e > best_edges:
                        best_edges = e
                        improved = True
                    else:
                        best_adj[i, j] = False
                        best_adj[j, i] = False
                else:
                    best_adj[i, j] = False
                    best_adj[j, i] = False
        
        # Try edge swaps: remove (i,j), add (k,l)
        for i, j in itertools.combinations(range(n), 2):
            if best_adj[i, j]:
                for k, l in itertools.combinations(range(n), 2):
                    if not best_adj[k, l] and (k, l) != (i, j):
                        best_adj[i, j] = False
                        best_adj[j, i] = False
                        best_adj[k, l] = True
                        best_adj[l, k] = True
                        if _is_k23_free(best_adj):
                            e = _count_edges(best_adj)
                            if e > best_edges:
                                best_edges = e
                                improved = True
                            else:
                                best_adj[i, j] = True
                                best_adj[j, i] = True
                                best_adj[k, l] = False
                                best_adj[l, k] = False
                        else:
                            best_adj[i, j] = True
                            best_adj[j, i] = True
                            best_adj[k, l] = False
                            best_adj[l, k] = False
        
        if not improved:
            break
    
    return best_adj

def construct():
    """Construct a K2,3-free graph on 20 vertices maximizing edges."""
    # Start with Paley-type construction
    adj = _paley_graph(20)
    
    # Optimize
    adj = _optimize(adj, max_iters=2000)
    
    # Convert to edge list
    edges = []
    for i in range(20):
        for j in range(i + 1, 20):
            if adj[i, j]:
                edges.append([i, j])
    
    return edges
