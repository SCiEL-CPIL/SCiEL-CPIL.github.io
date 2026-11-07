# EVOLVE-BLOCK-START
import time
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = P['n']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    vertices = list(range(n))
    
    # Helper: compute distinct degrees for a subset mask
    def _distinct_degrees(mask):
        deg_set = set()
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = _popcount(adj[v] & mask)
            deg_set.add(d)
            m ^= lsb
        return len(deg_set)
    
    # Helper: compute degree vector for a subset (for faster comparison)
    def _degree_vector(mask):
        degs = {}
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            degs[v] = _popcount(adj[v] & mask)
            m ^= lsb
        return degs
    
    best_degrees = 0
    best_mask = 0
    
    # Phase 1: Exhaustive search over all subsets with popcount >= min_verts
    # But we'll be smart: iterate by increasing size, and use pruning
    # Actually, for 17 vertices, 2^17 = 131072 subsets. Let's just do all.
    
    # Precompute popcount for all masks using dynamic programming
    popcounts = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcounts[i] = popcounts[i & (i - 1)] + 1
    
    # Iterate over all masks with popcount >= min_verts
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        if popcounts[mask] < min_verts:
            continue
        
        dc = _distinct_degrees(mask)
        if dc > best_degrees:
            best_degrees = dc
            best_mask = mask
            
            # If we've achieved the theoretical max for this subset size, 
            # we could potentially stop, but let's keep going for safety
            # Max distinct degrees for k vertices is k (all different)
            # But in a simple graph, degree range is [0, k-1], so max is k
            # If best_degrees == popcounts[mask], that's perfect for this size
            # But we want global max, so keep searching unless we hit n
    
    # Convert best_mask to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    
    # Ensure we have at least min_verts
    if len(result) < min_verts:
        # Shouldn't happen, but add more vertices if needed
        for v in vertices:
            if len(result) >= min_verts:
                break
            if v not in result:
                result.append(int(v))
    
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
