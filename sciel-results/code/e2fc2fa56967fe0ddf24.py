"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_c4(adj_matrix):
    """Count 4-cycles using common neighbor pairs.
    For each pair (u,v), if they share k common neighbors, contribute C(k,2).
    """
    n = adj_matrix.shape[0]
    total = 0
    # adj_matrix is 0/1 symmetric
    # common neighbors of u and v = sum of adj[u] & adj[v]
    for u in range(n):
        for v in range(u + 1, n):
            cn = np.sum(adj_matrix[u] & adj_matrix[v])
            if cn >= 2:
                total += cn * (cn - 1) // 2
    return total

def _count_c4_fast(adj_matrix):
    """Faster C4 count: sum over all pairs of C(common_neighbors, 2).
    Using matrix multiplication: (A^2)[u,v] = number of common neighbors of u and v.
    """
    n = adj_matrix.shape[0]
    # A^2 gives number of length-2 paths between each pair
    A2 = adj_matrix @ adj_matrix
    # For each pair (u,v), common neighbors = A2[u,v] (since no self-loops, diagonal doesn't matter)
    # C4 count = sum over u<v of C(A2[u,v], 2)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            cn = A2[u, v]
            if cn >= 2:
                total += cn * (cn - 1) // 2
    return total

def _count_c4_vec(adj_matrix):
    """Vectorized C4 count."""
    n = adj_matrix.shape[0]
    A2 = adj_matrix @ adj_matrix
    # Take upper triangle
    iu = np.triu_indices(n, k=1)
    cn = A2[iu]
    # C(cn, 2) = cn*(cn-1)//2, but only where cn >= 2
    mask = cn >= 2
    return np.sum(cn[mask] * (cn[mask] - 1) // 2)

def _random_graph(n, m, rng):
    """Generate random graph with n vertices and m edges."""
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = set()
    while len(edges) < m:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u, v] = 1
            adj[v, u] = 1
    return adj, list(edges)

def _edge_swap(adj, edges, u, v, x, y, rng):
    """Remove edge (u,v) and add edge (x,y). Return new adj, edges, and whether valid."""
    # Check (x,y) is not already an edge and x != y
    if x == y:
        return None
    if adj[x, y] == 1:
        return None
    # Remove (u,v)
    adj[u, v] = 0
    adj[v, u] = 0
    # Add (x,y)
    adj[x, y] = 1
    adj[y, x] = 1
    # Update edges set
    if u > v:
        u, v = v, u
    if x > y:
        x, y = y, x
    edges.remove((u, v))
    edges.add((x, y))
    return adj, edges

def construct():
    n = 31
    m = 91
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with random graph
    adj, edges = _random_graph(n, m, np_rng)
    best_adj = adj.copy()
    best_edges = list(edges)
    best_score = _count_c4_vec(adj)
    
    current_score = best_score
    current_adj = adj
    current_edges = edges
    
    # Simulated annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    max_iter = 200000
    
    iteration = 0
    while iteration < max_iter:
        if time.time() > deadline:
            break
        
        iteration += 1
        
        # Pick a random edge to remove
        idx = np_rng.randint(0, len(current_edges))
        u, v = current_edges[idx]
        
        # Pick a random non-edge to add
        x = np_rng.randint(0, n - 1)
        y = np_rng.randint(0, n - 1)
        if x == y or current_adj[x, y] == 1:
            continue
        
        # Perform swap
        old_adj = current_adj.copy()
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1
        
        # Update edges
        if u > v:
            u2, v2 = v, u
        else:
            u2, v2 = u, v
        if x > y:
            x2, y2 = y, x
        else:
            x2, y2 = x, y
        
        current_edges[idx] = (x2, y2)
        
        # Count C4
        new_score = _count_c4_vec(current_adj)
        delta = new_score - current_score
        
        if delta < 0 or np_rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_edges = list(current_edges)
        else:
            # Revert
            current_adj = old_adj
            current_edges[idx] = (u2, v2)
        
        T = max(T_min, T * cooling)
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
