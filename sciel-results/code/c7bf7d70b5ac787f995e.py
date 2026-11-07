P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # 1. Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # 2. Precompute is_clique and is_indep for all masks
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_valid = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    is_valid[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: rest must be clique, and v must connect to all in rest
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: rest must be independent, and v must not connect to any in rest
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True
            
        if is_clique[mask] or is_indep[mask]:
            is_valid[mask] = True
            
    # 3. DP to find minimum number of parts
    INF = 10**9
    dp = [INF] * size
    parent = [-1] * size
    chosen = [-1] * size
    
    dp[0] = 0
    
    for mask in range(1, size):
        # Iterate over all submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = prev
                    chosen[mask] = sub
            sub = (sub - 1) & mask
            
    # 4. Reconstruct the partition
    full_mask = size - 1
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = chosen[current_mask]
        # Extract vertices from sub
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask = parent[current_mask]
        
    # Convert to list of lists of ints
    # The order of parts doesn't matter, but vertices must be ints
    result = []
    for part in parts:
        result.append([int(v) for v in part])
        
    return result
