import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute valid subsets (cliques or independent sets)
    # We need to check all 2^n subsets
    valid_subs = []
    full_mask = (1 << n) - 1
    
    # To speed up checking, we can iterate subsets and check properties
    # But 16384 subsets is small enough to just check each one.
    
    for mask in range(1, full_mask + 1):
        is_clique = True
        is_indep = True
        
        # Check if clique
        # For each vertex in mask, all other vertices in mask must be neighbors
        # (mask & ~adj_mask[v] & ~(1<<v)) must be 0
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Check if any vertex in mask (excluding v) is NOT a neighbor
            # ~adj_mask[v] contains non-neighbors. 
            # We want to see if mask contains any non-neighbor of v.
            # Note: v is not a neighbor of itself in adj_mask.
            # So we check (mask & ~adj_mask[v]) & ~(1<<v) == 0
            if (mask & ~adj_mask[v] & ~(1 << v)) != 0:
                is_clique = False
                break
            m -= lsb
        
        if is_clique:
            valid_subs.append(mask)
            continue # No need to check indep if it's a clique? 
                     # Actually a set can be both (single vertex). 
                     # But for partitioning, we just need it to be valid.
                     # If it's a clique, it's valid. We don't need to add it again.
        
        if not is_clique:
            # Check if independent
            # For each vertex in mask, no other vertex in mask must be a neighbor
            # (mask & adj_mask[v]) must be 0
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if (mask & adj_mask[v]) != 0:
                    is_indep = False
                    break
                m -= lsb
            
            if is_indep:
                valid_subs.append(mask)
                
    # DP over subsets
    # dp[mask] = min number of parts to cover vertices in 'mask'
    size = 1 << n
    dp = [float('inf')] * size
    choice = [-1] * size
    dp[0] = 0
    
    # Iterate masks
    for mask in range(1, size):
        # Pick a pivot vertex to break symmetry
        # This ensures we don't consider permutations of the same partition
        # The pivot must be in the mask
        lsb = mask & -mask
        pivot = lsb.bit_length() - 1
        
        # Try all valid subsets that contain the pivot and are subsets of mask
        # Optimization: Pre-filter valid_subs? 
        # valid_subs is a list of masks.
        # We can just iterate through valid_subs and check conditions.
        # Number of valid_subs is likely small.
        
        best_val = float('inf')
        best_sub = -1
        
        for sub in valid_subs:
            # Check if sub is a subset of mask
            if (sub & mask) != sub:
                continue
            # Check if sub contains the pivot
            if not (sub & (1 << pivot)):
                continue
            
            new_mask = mask ^ sub
            val = 1 + dp[new_mask]
            if val < best_val:
                best_val = val
                best_sub = sub
        
        dp[mask] = best_val
        choice[mask] = best_sub
        
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == -1:
            # Should not happen if graph is valid
            break
        # Convert bitmask to list of vertices
        part = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m -= lsb
        partition.append(part)
        mask ^= sub
        
    # Ensure types are Python int and list
    # The vertices are already Python ints from bit_length
    # The partition is a list of lists of ints
    
    return partition

# EVOLVE-BLOCK-END
