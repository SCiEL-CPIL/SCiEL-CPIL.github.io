import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Precompute adjacency bitmasks for fast degree calculation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = 0
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # Iterate over all possible subsets
    # We can iterate by mask integer for speed
    for mask in range(1 << n):
        # Quick check: if time is running out, break
        if time.time() > deadline:
            break
            
        # Check size constraint
        # bin(mask).count('1') is slow, use bit_count if available (Python 3.10+) 
        # or a fallback. The environment likely has modern Python.
        # Let's use a helper or inline check.
        # Python 3.10+ has int.bit_count().
        if hasattr(mask, 'bit_count'):
            size = mask.bit_count()
        else:
            size = bin(mask).count('1')
            
        if size < min_v:
            continue
            
        # Calculate degrees for vertices in the subset
        degrees = set()
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph is popcount(adj[v] & mask)
            deg_mask = adj[v] & mask
            if hasattr(deg_mask, 'bit_count'):
                deg = deg_mask.bit_count()
            else:
                deg = bin(deg_mask).count('1')
            degrees.add(deg)
            m ^= lsb
            
        score = len(degrees)
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Early termination if we hit the theoretical max?
        # Max distinct degrees in a graph of size k is k.
        # If size == best_score, we can't do better for this size, but maybe a larger size has more?
        # Actually, distinct degrees <= size. So if best_score == size, it's perfect for that size.
        # But we want global max. Max possible is 17 (if we took all 17).
        # If best_score == n, we can stop.
        if best_score == n:
            break

    # Convert best_mask back to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    return result
