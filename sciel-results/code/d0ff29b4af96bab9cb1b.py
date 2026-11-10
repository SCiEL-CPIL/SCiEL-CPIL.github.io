import numpy as np
import time
import random

def _has_c6_fast(adj_mask, u, v):
    """
    Check if adding (u,v) creates a C6.
    adj_mask: list of int bitmasks for neighbors.
    """
    # Get neighbors as lists for faster iteration
    # Precompute neighbor lists from masks?
    # Or just iterate over bits.
    
    def get_neighbors(mask):
        neighbors = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neighbors.append(idx)
            m ^= lsb
        return neighbors

    neighbors_u = get_neighbors(adj_mask[u])
    neighbors_v = get_neighbors(adj_mask[v])
    
    # Paths of length 2 from u: (a, b)
    paths_u = []
    for a in neighbors_u:
        neighbors_a = get_neighbors(adj_mask[a])
        for b in neighbors_a:
            if b != a:
                paths_u.append((a, b))
                
    # Paths of length 2 from v: (d, c)
    paths_v = []
    for d in neighbors_v:
        neighbors_d = get_neighbors(adj_mask[d])
        for c in neighbors_d:
            if c != d:
                paths_v.append((d, c))
                
    # Check pairs
    for a, b in paths_u:
        for d, c in paths_v:
            # Need edge (b, c)
            if b == c:
                continue
            if not (adj_mask[b] & (1 << c)):
                continue
                
            # Check distinctness of u, a, b, c, d, v
            # u, v are distinct.
            # a, b, c, d must be distinct from u, v and each other.
            # a != u (true), a != v?
            # If a == v, then u-v is an edge in the path u-a-v...
            # But u-v is NOT in the graph. So a cannot be v.
            # Similarly d cannot be u.
            
            if a == v or d == u:
                continue
                
            if b == u or b == v or c == u or c == v:
                continue
                
            if a == b or a == c or a == d:
                continue
            if b == c:
                continue
            if c == d:
                continue
            if b == d:
                continue
                
            # If we passed all, they are distinct
            return True
            
    return False

def construct():
    n = 40
    start_time = time.time()
    timeout = 115
    
    adj = np.zeros((n, n), dtype=np.int8)
    adj_mask = [0] * n
    edges = []
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    
    candidates = list(all_edges)
    
    while candidates:
        if time.time() - start_time > timeout:
            break
            
        batch_size = min(100, len(candidates))
        batch = candidates[:batch_size]
        candidates = candidates[batch_size:]
        
        valid_edges_in_batch = []
        
        for u, v in batch:
            if _has_c6_fast(adj_mask, u, v):
                continue
            valid_edges_in_batch.append((u, v))
            
        if not valid_edges_in_batch:
            continue
            
        # Pick best by spectral gap
        # If many valid, sample
        if len(valid_edges_in_batch) > 5:
            valid_edges_in_batch = random.sample(valid_edges_in_batch, 5)
            
        best_edge = None
        best_gap = -1e9
        
        for u, v in valid_edges_in_batch:
            A = adj.copy()
            A[u][v] = 1
            A[v][u] = 1
            try:
                eigvals = np.linalg.eigvalsh(A)
                gap = eigvals[-1] - eigvals[-2]
            except:
                gap = 0.0
                
            if gap > best_gap:
                best_gap = gap
                best_edge = (u, v)
                
        if best_edge is None:
            continue
            
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
    return edges
