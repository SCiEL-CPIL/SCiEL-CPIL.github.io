import sys
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_ind for all masks
    size = 1 << n
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        # rest must be a clique and rest must be subset of adj[v]
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        # Check independent set
        # rest must be an independent set and rest must have no intersection with adj[v]
        if is_ind[rest] and (rest & adj[v]) == 0:
            is_ind[mask] = True
            
    # Precompute is_valid
    is_valid = [is_clique[i] or is_ind[i] for i in range(size)]
    
    # DP
    INF = float('inf')
    dp = [INF] * size
    parent = [None] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Find lowest set bit index
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        best_val = INF
        best_sub = None
        
        # Iterate submasks of mask
        sub = mask
        while sub:
            # Optimization: We only care about submasks that contain v
            # and are valid.
            if (sub & lsb) and is_valid[sub]:
                rem = mask ^ sub
                val = dp[rem] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        parent[mask] = best_sub
        
    # Reconstruct partition
    full_mask = size - 1
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub is None:
            break # Should not happen if graph is valid
        # Extract vertices from sub
        part_vertices = []
        temp = sub
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            temp ^= lsb
        parts.append(part_vertices)
        mask ^= sub
        
    return parts

# EVOLVE-BLOCK-END
