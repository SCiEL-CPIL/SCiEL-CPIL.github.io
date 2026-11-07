"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    min_verts = P['minimum_vertices']

    # Precompute neighbor bitmask for each vertex
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_score = 0
    best_mask = 0

    # Enumerate all 2^n masks
    total_masks = 1 << n
    # Precompute list of set bits for each mask? No, too memory intensive.
    # Instead, iterate and use bit tricks.

    # For efficiency, we'll iterate over all masks and skip those with < min_verts bits
    # We can use a recursive or iterative approach.
    
    # Direct iteration over all masks:
    for mask in range(total_masks):
        if time.time() > deadline:
            break
        # Count bits in mask
        pop = mask.bit_count()
        if pop < min_verts:
            continue
        
        # Compute degrees for each vertex in the mask
        # For each vertex v in mask, degree = (adj_mask[v] & mask).bit_count()
        # We need to collect all degrees and count distinct ones
        
        # To avoid creating a set every time (expensive), we can use a small array
        # Max possible degree in induced subgraph is pop-1 (at most 16)
        # Use a boolean array of size 17 to track which degrees appear
        
        deg_seen = 0  # bitmask of which degrees have been seen (degree d -> bit d)
        
        m = mask
        while m:
            # Extract lowest set bit
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj_mask[v] & mask).bit_count()
            deg_seen |= (1 << deg)
            m ^= lb
        
        distinct = deg_seen.bit_count()
        if distinct > best_score:
            best_score = distinct
            best_mask = mask
            # If we've achieved the maximum possible (equal to pop), we can't do better
            # for this size, but a larger subset might have more distinct degrees
            # Actually, max distinct degrees for a subset of size k is k (degrees 0..k-1)
            # So if distinct == pop, that's the max for this size
            # But a larger subset could have more distinct degrees
            # So we continue unless we hit a theoretical upper bound
            # Theoretical max is n (17) but we need at least 8 vertices
            # If best_score == n, we can stop
            if best_score >= n:
                break
    
    # Extract vertices from best_mask
    if best_mask == 0:
        # Fallback: take first 8 vertices
        best_mask = (1 << min_verts) - 1
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(int(v))
        m ^= lb
    
    # Ensure at least 8 vertices (should be guaranteed by our search)
    if len(result) < min_verts:
        # Fallback
        result = [int(i) for i in range(min_verts)]
    
    return result
# EVOLVE-BLOCK-END
