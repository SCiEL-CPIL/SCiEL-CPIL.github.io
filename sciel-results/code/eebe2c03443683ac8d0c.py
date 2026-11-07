"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin out of 60s

    n = P['n']
    edges = P['edges']
    min_k = P['minimum_vertices']

    # Build adjacency bitmask for each vertex
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_subset_mask = 0
    max_distinct = 0

    # Iterate over all possible subsets (bitmasks)
    # Total 2^17 = 131072 subsets
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Check size constraint
        k = mask.bit_count()
        if k < min_k:
            continue

        # Compute degrees for vertices in the subset
        degrees = set()
        # Iterate through vertices in the subset
        m = mask
        while m:
            # Get the lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Compute degree of v in induced subgraph
            deg = (adj_mask[v] & mask).bit_count()
            degrees.add(deg)
            m ^= lsb
            
        distinct_count = len(degrees)
        
        if distinct_count > max_distinct:
            max_distinct = distinct_count
            best_subset_mask = mask
            
            # Early exit if we reached the theoretical maximum for this size
            # Note: We are iterating masks in increasing order of value, not size.
            # However, if we find k distinct degrees, that's max for size k.
            # But a larger size k' could have k' > k distinct degrees.
            # So we cannot simply exit on distinct_count == k unless we know no larger k exists with higher distinct count.
            # Since we are doing a full exhaustive search, we should just finish.
            # But 131k iterations is very fast. No early exit needed for correctness, 
            # and it's hard to justify early exit without complex tracking.
            
    # Convert best_subset_mask to list of vertices
    best_subset = []
    m = best_subset_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_subset.append(v)
        m ^= lsb

    # Ensure types are correct (Python int, list)
    return list(best_subset)
# EVOLVE-BLOCK-END
