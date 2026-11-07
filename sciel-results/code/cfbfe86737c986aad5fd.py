P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute which subsets are cliques and independent sets
    # A subset is a clique if every pair has an edge
    # A subset is independent if no pair has an edge
    
    num_subsets = 1 << n
    
    # is_clique[mask] = True if mask is a clique
    # is_independent[mask] = True if mask is an independent set
    is_clique = [False] * num_subsets
    is_independent = [False] * num_subsets
    is_valid = [False] * num_subsets
    
    # Empty set and singletons are both cliques and independent
    for mask in range(num_subsets):
        # Check if mask is a clique
        clique = True
        independent = True
        if mask != 0:
            # Get the lowest set bit
            v = (mask & (-mask)).bit_length() - 1
            rest = mask & ~(1 << v)
            if rest != 0:
                # Check if v is connected to all in rest (for clique)
                # and not connected to any in rest (for independent)
                for u_bit in range(n):
                    if rest & (1 << u_bit):
                        if not (adj[v] & (1 << u_bit)):
                            clique = False
                            break
                if clique:
                    # v is connected to all in rest, need rest to be clique too
                    if not is_clique[rest]:
                        clique = False
                else:
                    # Check independent
                    for u_bit in range(n):
                        if rest & (1 << u_bit):
                            if (adj[v] & (1 << u_bit)):
                                independent = False
                                break
                    if independent:
                        if not is_independent[rest]:
                            independent = False
            
            if not clique and not independent:
                # Re-check more carefully
                clique = True
                independent = True
                vertices = [i for i in range(n) if mask & (1 << i)]
                if len(vertices) > 1:
                    for i in range(len(vertices)):
                        for j in range(i+1, len(vertices)):
                            u, w = vertices[i], vertices[j]
                            if adj[u] & (1 << w):
                                independent = False
                            else:
                                clique = False
                    # If we broke early, need to re-evaluate
                    # Actually let me just do a clean check
        
        # Clean check for small n
        vertices = [i for i in range(n) if mask & (1 << i)]
        if len(vertices) <= 1:
            is_clique[mask] = True
            is_independent[mask] = True
        else:
            clique = True
            independent = True
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u, w = vertices[i], vertices[j]
                    has_edge = bool(adj[u] & (1 << w))
                    if has_edge:
                        independent = False
                    else:
                        clique = False
                    if not clique and not independent:
                        break
                if not clique and not independent:
                    break
            is_clique[mask] = clique
            is_independent[mask] = independent
        
        is_valid[mask] = is_clique[mask] or is_independent[mask]
    
    # DP: dp[mask] = min number of valid subsets to partition mask
    # dp[0] = 0
    # dp[mask] = min over valid submasks S of mask (S != 0): dp[mask ^ S] + 1
    
    import math
    INF = float('inf')
    dp = [INF] * num_subsets
    dp[0] = 0
    
    # For each mask, try all valid submasks
    # Optimization: iterate submasks of mask
    for mask in range(1, num_subsets):
        # Enumerate all submasks of mask
        sub = mask
        best = INF
        while sub > 0:
            if is_valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
            sub = (sub - 1) & mask
        dp[mask] = best
    
    # Full mask
    full_mask = (1 << n) - 1
    min_parts = dp[full_mask]
    
    # Reconstruct the partition
    # We need to find the actual partition, not just the count
    # Use backtracking with the dp table
    
    partition = []
    mask = full_mask
    while mask > 0:
        # Find a valid submask S of mask such that dp[mask] = dp[mask ^ S] + 1
        sub = mask
        found = False
        while sub > 0:
            if is_valid[sub] and dp[mask ^ sub] + 1 == dp[mask]:
                # Add this submask as a part
                part = [i for i in range(n) if sub & (1 << i)]
                partition.append(part)
                mask = mask ^ sub
                found = True
                break
            sub = (sub - 1) & mask
        if not found:
            # Should not happen, but fallback
            part = [i for i in range(n) if mask & (1 << i)]
            partition.append(part)
            mask = 0
    
    return partition
