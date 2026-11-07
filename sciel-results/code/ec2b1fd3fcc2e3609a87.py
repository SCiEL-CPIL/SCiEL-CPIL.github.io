"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Precompute adjacency bitmasks for O(1) degree calculation via bit_count
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    best_count = 0
    best_mask = 0
    
    # Iterate over all 2^n subsets
    # Use bit_count if available, else fallback
    try:
        (1).bit_count()
        use_bit_count = True
    except AttributeError:
        use_bit_count = False
        # Fallback popcount
        def popcount(x):
            return bin(x).count('1')
    
    for mask in range(1 << n):
        # Check minimum vertex constraint
        if use_bit_count:
            size = mask.bit_count()
        else:
            size = popcount(mask)
            
        if size < min_v:
            continue
            
        # Calculate degrees for vertices in the induced subgraph
        degrees = []
        for v in range(n):
            if (mask >> v) & 1:
                # Intersection of adjacency of v and the subset mask
                # Count bits gives the degree in the induced subgraph
                if use_bit_count:
                    deg = (adj_masks[v] & mask).bit_count()
                else:
                    deg = popcount(adj_masks[v] & mask)
                degrees.append(deg)
                
        distinct = len(set(degrees))
        
        if distinct > best_count:
            best_count = distinct
            best_mask = mask
            # Theoretical max for a graph of size 'size' is 'size'
            # If we reach 'size', we can't do better for this specific size.
            # However, larger sizes can have more distinct degrees.
            # If distinct == n, we found the global max, break.
            if distinct == n:
                break
        # Pruning: If best_count is already >= size, we can't improve with this size.
        # But we don't know if a larger size will improve.
        # We can't break the outer loop easily based on size unless we iterate by size.
        # Given n=17, full scan is fast enough.
        
    # Extract vertices from best_mask
    result = []
    for v in range(n):
        if (best_mask >> v) & 1:
            result.append(int(v))
            
    return result
# EVOLVE-BLOCK-END
