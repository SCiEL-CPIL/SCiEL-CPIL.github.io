"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    deadline = time.monotonic() + 85.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute degree of each vertex within every possible mask
    # deg_in_mask[v][mask] = number of neighbors of v that are in mask
    # This is 17 * 2^17 = 1,782,576 entries, which is manageable
    # But actually, we can compute on the fly since bit_count is fast.
    
    # Better approach: precompute for each vertex v, the list of neighbors
    # Then for a given mask, degree of v = popcount(adj[v] & mask)
    
    # Let's try a hybrid approach:
    # 1. First, do a quick random/local search to find a good initial solution
    # 2. Then do exhaustive search with early termination
    
    best_deg = 0
    best_vertices = list(range(min_size))
    
    # Quick heuristic: try all subsets of size exactly min_size first
    # This is C(17,8) = 24310 subsets, very fast
    for combo in itertools.combinations(range(n), min_size):
        if time.monotonic() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = set()
        for v in combo:
            degs.add((adj[v] & mask).bit_count())
        nd = len(degs)
        if nd > best_deg:
            best_deg = nd
            best_vertices = list(combo)
            if best_deg == min_size:
                break
    
    # If we already found the maximum possible for size 8, we're done
    if best_deg == min_size:
        return [int(v) for v in best_vertices]
    
    # Now try larger sizes, but with a smarter approach
    # For each size from min_size+1 to n, enumerate subsets
    # Use early termination if we can't beat the current best
    for size in range(min_size + 1, n + 1):
        if time.monotonic() > deadline:
            break
        
        # The maximum possible distinct degrees for a subset of size k is k
        # If our current best is already >= size, we can't improve by going to larger sizes
        # Wait, that's not right. We want to maximize distinct degrees, and for size k, max is k.
        # So if best_deg >= size, we might still find a subset of size k with k distinct degrees
        # which could be better than best_deg if k > best_deg.
        
        # Actually, let's just enumerate and check.
        # But we can add a pruning: if size <= best_deg, we can't beat it
        # because max distinct degrees for size k is k, so if k <= best_deg, no improvement possible.
        if size <= best_deg:
            continue
        
        for combo in itertools.combinations(range(n), size):
            if time.monotonic() > deadline:
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            degs = set()
            for v in combo:
                degs.add((adj[v] & mask).bit_count())
            
            nd = len(degs)
            if nd > best_deg:
                best_deg = nd
                best_vertices = list(combo)
                if best_deg == size:
                    break
        else:
            continue
        break
    
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
