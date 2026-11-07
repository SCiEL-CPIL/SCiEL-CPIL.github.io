import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    start = time.time()
    deadline = start + 50.0

    # Build adjacency as bitmask
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    best_count = -1
    best_verts = []

    # Iterate subset sizes from n down to minv
    # If we find a subset of size k with k distinct degrees, 
    # and we are iterating down, any smaller subset size j < k 
    # can have at most j distinct degrees, which is < k.
    # So if best_count == k, we can stop.
    
    for k in range(n, minv - 1, -1):
        # If we already have a score >= k, no smaller subset can beat it
        if k <= best_count:
            break
        
        # Iterate all combinations of size k
        # itertools.combinations is efficient in C
        for comb in itertools.combinations(range(n), k):
            # Time guard
            if time.time() > deadline:
                return best_verts if best_verts else list(range(minv))
            
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            degs = set()
            # Compute degrees for each vertex in the subset
            for v in comb:
                d = (adj[v] & mask).bit_count()
                degs.add(d)
                
                # Early break if we've found k distinct degrees (max possible for size k)
                if len(degs) == k:
                    break
            
            nd = len(degs)
            if nd > best_count:
                best_count = nd
                best_verts = list(comb)
                
                # If we achieved the theoretical maximum for this size k
                if nd == k:
                    # We can't do better than k for this size.
                    # Since we are checking smaller sizes next, and they can't exceed their size,
                    # and k is the current size, any smaller size j < k has max distinct j < k.
                    # So we can stop the entire search.
                    return best_verts

    # Fallback if loop finished without returning (should not happen if minv is valid)
    if not best_verts:
        best_verts = list(range(minv))
        
    return best_verts
# EVOLVE-BLOCK-END
