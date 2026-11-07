"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _distinct_degrees(subset, adj):
    """Compute number of distinct degrees in induced subgraph on subset."""
    subset_set = set(subset)
    degrees = set()
    for v in subset:
        deg = sum(1 for u in adj[v] if u in subset_set)
        degrees.add(deg)
    return len(degrees)

def _popcount(x):
    return bin(x).count('1')

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency masks for fast bit operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_score = -1
    best_mask = 0
    
    # Precompute popcount for all possible 17-bit masks
    # Use dynamic programming: popcount[i] = popcount[i >> 1] + (i & 1)
    max_mask = 1 << n
    popcount_table = [0] * max_mask
    for i in range(1, max_mask):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    # Iterate over all possible subsets using bitmasks
    for mask in range(1, max_mask):
        k = popcount_table[mask]
        if k < min_verts:
            continue
        
        # Upper bound: can't have more distinct degrees than k (or k-1 since max degree is k-1)
        # Actually max distinct degrees is k (degrees 0 through k-1, that's k values)
        if k <= best_score:
            continue
        
        degrees = set()
        for v in range(n):
            if (mask >> v) & 1:
                deg = popcount_table[adj_mask[v] & mask]
                degrees.add(deg)
        
        score = len(degrees)
        if score > best_score:
            best_score = score
            best_mask = mask
            # If we achieved the theoretical maximum for this size, we can't do better
            # But we might find a larger subset with more distinct degrees
    
    # Convert best mask to list of vertices
    result = [i for i in range(n) if (best_mask >> i) & 1]
    
    if not result:
        return list(range(min_verts))
        
    return result
# EVOLVE-BLOCK-END
