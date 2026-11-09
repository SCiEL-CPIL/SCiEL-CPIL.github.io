"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']

    # Precompute neighbor masks for O(1) degree queries
    neighbor_masks = [0] * n
    for u, v in edges:
        neighbor_masks[u] |= (1 << v)
        neighbor_masks[v] |= (1 << u)

    total_masks = 1 << n
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Precompute popcount lookup for all 2^17 values
    popcount_table = np.zeros(total_masks, dtype=np.uint8)
    for i in range(1, total_masks):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)

    # Precompute: for each vertex v, an array of length total_masks
    # where deg_table[v][mask] = popcount(mask & neighbor_masks[v])
    # Shape: (n, total_masks), dtype uint8
    deg_table = np.zeros((n, total_masks), dtype=np.uint8)
    nm_arr = np.array(neighbor_masks, dtype=np.uint32)
    all_masks_arr = np.arange(total_masks, dtype=np.uint32)
    
    for v in range(n):
        # degree of v in induced subgraph of mask = popcount(mask & neighbor_masks[v])
        and_vals = np.bitwise_and(all_masks_arr, nm_arr[v])
        # Use popcount_table via fancy indexing
        deg_table[v] = popcount_table[and_vals]

    # Now for each mask, we need to count distinct degrees among vertices in the mask
    # Strategy: for each mask, collect degrees of all vertices in the mask, count distinct
    # This is still O(n) per mask, but with vectorized precomputation
    
    best_score = -1
    best_mask = 0

    # Process masks in batches using vectorized operations where possible
    # For each mask, we need: set of {deg_table[v][mask] for v in mask}
    # Let's iterate but use the precomputed table for O(1) degree lookup
    
    for mask in range(total_masks):
        if (mask & 1023) == 0 and time.time() - start_time > deadline:
            break

        size = int(popcount_table[mask])
        
        if size < min_vertices:
            continue
            
        if size <= best_score:
            continue

        # Compute distinct degrees using precomputed table
        deg_set = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = int(deg_table[v][mask])
            deg_set.add(deg)
            m -= lsb

        score = len(deg_set)
        if score > best_score:
            best_score = score
            best_mask = mask
            if best_score == n:
                break

    if best_mask == 0:
        return [int(i) for i in range(min_vertices)]

    vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(v)
        m -= lsb

    return [int(v) for v in vertices]
# EVOLVE-BLOCK-END
