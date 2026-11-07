# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Fast popcount
    try:
        # Python 3.10+
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')
    
    best_score = 0
    best_mask = 0
    
    # Iterate over all masks
    # 2^17 = 131072 iterations
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        
        # Check size
        if popcount(mask) < min_vertices:
            continue
            
        # Compute degrees
        # We need to check vertices that are in the mask.
        # Iterate through set bits of mask.
        degrees = set()
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            # Index of lsb
            v = (lsb.bit_length() - 1)
            # Degree of v in mask is popcount(adj_mask[v] & mask)
            deg = popcount(adj_mask[v] & mask)
            degrees.add(deg)
            m ^= lsb
            
        distinct = len(degrees)
        
        if distinct > best_score:
            best_score = distinct
            best_mask = mask
            
            # Max possible distinct degrees for a graph of size k is k-1.
            # If we reach n-1, we can't do better.
            if best_score >= n - 1:
                break
    
    # Convert best_mask to list of vertices
    if best_mask == 0:
        # Fallback if no mask found (should not happen)
        return [int(i) for i in range(min_vertices)]
        
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = (lsb.bit_length() - 1)
        best_vertices.append(v)
        m ^= lsb
        
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
