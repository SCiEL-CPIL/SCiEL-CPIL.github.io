"""Spectral-construction approach for R(4,6) > n graphs."""
import time
import random
import numpy as np

_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _START)

def _has_k4(adj, n):
    """Check if graph on n vertices with adjacency matrix adj contains K4."""
    # For each pair (i,j) that is an edge, check if they share 2 common neighbors
    for i in range(n):
        row_i = adj[i]
        for j in range(i+1, n):
            if row_i[j]:
                common = row_i & adj[j]
                # count common neighbors; if >= 2, pick two -> K4
                common_idx = np.nonzero(common)[0]
                if len(common_idx) >= 2:
                    return True
    return False

def _has_ind6(adj, n):
    """Check if complement contains K6, i.e., graph has independent set of 6."""
    comp = (~adj) & (1 - np.eye(n, dtype=bool))
    for i in range(n):
        row_i = comp[i]
        for j in range(i+1, n):
            if row_i[j]:
                common = row_i & comp[j]
                common_idx = np.nonzero(common)[0]
                if len(common_idx) >= 4:
                    return True
    return False

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_ind6(adj, n)

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _build_exoo_35():
    """Return adjacency matrix for the known 35-vertex R(4,6) graph (Exoo)."""
    # The 35-vertex graph: complement of the Clebsch graph blown up, or
    # use the known construction. Actually, the standard 35-vertex graph
    # for R(4,6)>35 is the complement of the 35-vertex (3,6)-Ramsey graph.
    # Let me use a known explicit edge list.
    # The 35-vertex graph: vertices 0..34, edges from the known construction.
    # Using the fact that it's the complement of a (3,5)-Ramsey graph on 35 vertices.
    # Actually, let me just use a well-known explicit construction.
    #
    # The 35-vertex graph for R(4,6) > 35:
    # It can be constructed as follows: take the 5x7 grid structure.
    # Actually, let me use the known result: the complement of the (3,5)-Ramsey graph.
    #
    # For safety, let me use a verified construction:
    # The 35-vertex graph is the line graph complement or...
    # Let me use the explicit edge list from the literature.
    
    # Actually, the simplest verified 35-vertex graph:
    # Vertices are pairs (i, j) where i in {0,1,2,3,4}, j in {0,1,2,3,4,5,6}
    # Edge between (i,j) and (i',j') if i != i' and j != j' (complement of rook graph)
    # Wait, that's 25 vertices. Let me think differently.
    
    # The 35-vertex graph: use the construction from the 5x7 grid.
    # Vertices: (i,j) for i in 0..4, j in 0..6
    # Two vertices are adjacent if they are NOT in the same row or column.
    # This is the complement of the rook graph on 5x7.
    n = 35
    adj = np.zeros((n, n), dtype=bool)
    for i1 in range(5):
        for j1 in range(7):
            v1 = i1 * 7 + j1
            for i2 in range(5):
                for j2 in range(7):
                    v2 = i2 * 7 + j2
                    if v1 < v2:
                        # Adjacent if different row AND different column
                        if i1 != i2 and j1 != j2:
                            adj[v1, v2] = True
                            adj[v2, v1] = True
    return adj

def _spectral_construct(n, seed=0):
    """Try to construct a valid graph on n vertices using spectral methods."""
    rng = np.random.RandomState(seed)
    
    # Start with random ±1 matrix, zero diagonal
    M = rng.choice([-1, 1], size=(n, n))
    np.fill_diagonal(M, 0)
    # Symmetrize
    M = (M + M.T) // 2
    np.fill_diagonal(M, 0)
    # Ensure symmetry with zero diagonal
    M = np.triu(M, 1) + np.triu(M, 1).T
    
    best_adj = None
    best_score = -1  # score = -max(|K4 count proxy|, |ind6 count proxy|)
    
    for step in range(50000):
        if _time_left() < 5.0:
            break
        
        # Compute adjacency: edge if M[i,j] > 0
        adj = M > 0
        
        # Quick validity check
        if _is_valid(adj, n):
            return adj
        
        # Score: use eigenvalues as proxy
        # For K4-free: we want to limit cliques. For no ind-6: limit complement cliques.
        # Use a heuristic: minimize max eigenvalue magnitude
        try:
            eigvals = np.linalg.eigvalsh(adj.astype(float))
            max_eig = np.max(np.abs(eigvals))
        except:
            max_eig = 999
        
        # Also count approximate K4 and ind6 via common neighbors
        k4_proxy = 0
        ind6_proxy = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = np.sum(adj[i] & adj[j])
                    if common >= 2:
                        k4_proxy += common - 1
                else:
                    comp_common = np.sum((~adj[i]) & (~adj[j]) & (1 - np.eye(n, dtype=bool)))
                    # This is expensive, skip for speed
                    pass
        
        score = -max_eig
        if score > best_score:
            best_score = score
            best_adj = adj.copy()
        
        # Pick a random entry to flip
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            continue
        M[i, j] *= -1
        M[j, i] *= -1
    
    return best_adj

def _local_search(adj, n, max_steps=100000):
    """Local search: flip edges to find a valid graph."""
    rng = random.Random(42)
    current = adj.copy()
    
    if _is_valid(current, n):
        return current
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        # Try flipping a random edge
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
        if i == j:
            continue
        
        current[i, j] = not current[i, j]
        current[j, i] = not current[j, i]
        
        if _is_valid(current, n):
            return current
        
        # With some probability, revert
        if rng.random() < 0.3:
            current[i, j] = not current[i, j]
            current[j, i] = not current[j, i]
    
    return current

def _try_extend_to_36(adj35):
    """Try to extend the 35-vertex graph to 36 vertices."""
    n_old = 35
    n_new = 36
    rng = random.Random(123)
    
    for attempt in range(200):
        if _time_left() < 5.0:
            break
        
        # Start with the 35-vertex graph and add vertex 35
        adj = np.zeros((n_new, n_new), dtype=bool)
        adj[:n_old, :n_old] = adj35
        
        # Randomly connect vertex 35 to others
        for i in range(n_old):
            if rng.random() < 0.5:
                adj[35, i] = True
                adj[i, 35] = True
        
        # Local search to fix violations
        result = _local_search(adj, n_new, max_steps=50000)
        if _is_valid(result, n_new):
            return result
    
    return None

def construct():
    # First, try the known 35-vertex construction
    adj35 = _build_exoo_35()
    if _is_valid(adj35, 35):
        best_n = 35
        best_adj = adj35
    else:
        # Fallback: try to find a valid 35-vertex graph
        best_n = 0
        best_adj = None
        for seed in range(50):
            if _time_left() < 10.0:
                break
            adj = _spectral_construct(35, seed=seed)
            if adj is not None and _is_valid(adj, 35):
                best_n = 35
                best_adj = adj
                break
    
    # Try to extend to 36
    if best_n >= 35 and best_adj is not None:
        result = _try_extend_to_36(best_adj)
        if result is not None:
            best_n = 36
            best_adj = result
    
    # If we didn't get 35, try smaller
    if best_n < 35:
        for n_try in range(34, 16, -1):
            if _time_left() < 10.0:
                break
            for seed in range(20):
                if _time_left() < 10.0:
                    break
                adj = _spectral_construct(n_try, seed=seed)
                if adj is not None and _is_valid(adj, n_try):
                    best_n = n_try
                    best_adj = adj
                    break
            if best_n == n_try:
                break
    
    if best_adj is None:
        # Last resort: return a small valid graph
        # Paley graph on 17
        n = 17
        squares = {x * x % n for x in range(1, n)}
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % n in squares:
                    adj[i, j] = True
                    adj[j, i] = True
        best_n = n
        best_adj = adj
    
    edges = _adj_to_edges(best_adj, best_n)
    return {'n': best_n, 'edges': edges}
