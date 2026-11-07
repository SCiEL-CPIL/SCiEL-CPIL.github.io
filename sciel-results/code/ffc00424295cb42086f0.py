"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute degree in full graph for each vertex
    full_deg = [adj_mask[i].bit_count() for i in range(n)]
    
    def count_distinct_degrees(mask, k):
        """Count distinct degrees in induced subgraph given as bitmask of size k."""
        degs = []
        for i in range(n):
            if mask & (1 << i):
                degs.append((adj_mask[i] & mask).bit_count())
        return len(set(degs))
    
    best_degrees = 0
    best_subset = None
    best_result = list(range(min_verts))  # fallback
    
    # Strategy: For 8-vertex subsets, max possible distinct degrees is 8 (values 0-7)
    # Search size 8 first (C(17,8)=24310 subsets), then larger sizes if time permits
    # Use early termination: if we find 8 distinct degrees in an 8-vertex subset, that's optimal
    
    # Phase 1: Exhaustive search over all 8-vertex subsets
    for subset in combinations(range(n), min_verts):
        if time.time() > deadline - 2.0:
            break
        
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        nd = count_distinct_degrees(mask, min_verts)
        if nd > best_degrees:
            best_degrees = nd
            best_subset = subset
            best_result = list(subset)
            
        # If we achieved the theoretical maximum for 8 vertices, we can stop
        if best_degrees >= min_verts:
            break
    
    # Phase 2: If time remains, try larger subsets (9, 10, ..., 17)
    # For larger subsets, we can potentially get more distinct degree values
    for k in range(min_verts + 1, n + 1):
        if time.time() > deadline - 2.0:
            break
        
        for subset in combinations(range(n), k):
            if time.time() > deadline - 2.0:
                break
            
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            nd = count_distinct_degrees(mask, k)
            if nd > best_degrees:
                best_degrees = nd
                best_subset = subset
                best_result = list(subset)
    
    if best_subset is not None:
        return [int(v) for v in best_subset]
    return best_result

# EVOLVE-BLOCK-END
