import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    best_score = 0
    best_mask = 0
    
    # Max possible distinct degrees is 9 (0 to 8) since max degree is 8.
    # We iterate through all possible subsets.
    # 2^17 = 131072 iterations, which is very fast.
    
    for mask in range(1 << n):
        # Check if number of vertices is at least 8
        # bin(mask).count("1") is slow, use a precomputed table or bit_count if available
        # Python 3.10+ has int.bit_count()
        if hasattr(int, 'bit_count'):
            k = mask.bit_count()
        else:
            k = bin(mask).count('1')
            
        if k < 8:
            continue
            
        degrees = set()
        for v in range(n):
            if mask & (1 << v):
                # Count neighbors in the subset
                deg = (adj[v] & mask).bit_count() if hasattr(int, 'bit_count') else bin(adj[v] & mask).count('1')
                degrees.add(deg)
                
        distinct_count = len(degrees)
        if distinct_count > best_score:
            best_score = distinct_count
            best_mask = mask
            
            if best_score == 9:
                # Theoretical maximum
                break
                
    # Convert best_mask to list of vertices
    best_subset = [v for v in range(n) if best_mask & (1 << v)]
    
    return best_subset
