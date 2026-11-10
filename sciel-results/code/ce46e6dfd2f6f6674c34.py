import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute clique and independent set status for all masks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        # Find the lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True
            
    # valid[mask] is True if mask is a valid part (clique or independent)
    valid = [False] * size
    for m in range(size):
        valid[m] = is_clique[m] or is_indep[m]
        
    # DP: dp[mask] = min number of parts to partition mask
    # choice[mask] = the part containing the lowest set bit of mask
    INF = float('inf')
    dp = [INF] * size
    choice = [0] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Check time occasionally
        if time.time() > deadline:
            # Fallback: greedy or trivial partition if DP is too slow (unlikely for n=14)
            # But for safety, we can just return a valid partition found so far or fallback to singletons
            # Since n=14, this loop finishes in ms. This check is purely defensive.
            pass
            
        lsb = mask & -mask
        # Iterate over all submasks 'sub' of 'mask' that contain 'lsb'
        # Let rest = mask ^ lsb
        rest = mask ^ lsb
        
        best_cost = INF
        best_sub = 0
        
        # sub_rest iterates over all submasks of rest
        sub_rest = rest
        while True:
            sub = lsb | sub_rest
            if valid[sub]:
                remaining = mask ^ sub
                cost = dp[remaining]
                if cost < best_cost:
                    best_cost = cost
                    best_sub = sub
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest
        
        dp[mask] = best_cost + 1
        choice[mask] = best_sub
        
    # Reconstruct partition
    if dp[full_mask] >= INF:
        # Should not happen, but fallback
        return [[i] for i in range(n)]
        
    partition = []
    mask = full_mask
    while mask != 0:
        sub = choice[mask]
        # Convert mask to list of vertices
        part_verts = []
        temp = sub
        while temp:
            lsb_v = temp & -temp
            v = lsb_v.bit_length() - 1
            part_verts.append(v)
            temp ^= lsb_v
        partition.append(part_verts)
        mask ^= sub
        
    # Ensure types are correct: list of lists of Python ints
    final_result = []
    for part in partition:
        final_result.append([int(v) for v in part])
        
    return final_result
