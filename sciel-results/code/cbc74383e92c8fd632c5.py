import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute valid subsets
    # valid[S] is True if S is a clique or an independent set
    valid = [False] * (1 << n)
    valid[0] = True # Empty set is valid for DP base, though not used as a part
    
    # Helper to get least significant bit index
    # Python 3.11+ has int.bit_count etc, but let's stick to standard.
    # lsb = x & -x, index = (lsb.bit_length() - 1)
    
    for S in range(1, 1 << n):
        # Find least significant bit
        lsb = S & -S
        u = lsb.bit_length() - 1
        rest = S ^ lsb
        
        # Check if clique
        # All vertices in 'rest' must be neighbors of u
        # adj_mask[u] contains neighbors of u
        # If (rest & ~adj_mask[u]) == 0, then all in rest are neighbors of u
        # Since we are building up, we assume rest is already a clique? 
        # Wait, the definition of clique requires ALL pairs.
        # If rest is a clique, and u is connected to all in rest, then S is a clique.
        # So we need is_clique[rest] to be True?
        # Yes, recursive definition.
        # is_clique[S] = is_clique[rest] AND (rest subset of adj[u])
        
        # Similarly for independent set
        # is_independent[S] = is_independent[rest] AND (rest subset of non_adj[u])
        
        # We can compute these in increasing order of S.
        
        # Check Clique condition
        # (rest & ~adj_mask[u]) == 0
        if (rest & ~adj_mask[u]) == 0:
             # This is true if rest is a clique AND u connects to all in rest
             # But we need to know if rest is a clique.
             # Let's maintain separate arrays or just check validity of rest.
             # Actually, if rest is a clique, and u connects to all, S is clique.
             # If rest is NOT a clique, S cannot be a clique regardless of u.
             # So: is_clique[S] = is_clique[rest] and ((rest & ~adj_mask[u]) == 0)
             
             pass # Logic needs to be inside the loop with stored states
        
    # Let's use two boolean arrays for clarity and speed
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for S in range(1, 1 << n):
        lsb = S & -S
        u = lsb.bit_length() - 1
        rest = S ^ lsb
        
        # Clique check
        # u must be connected to all vertices in rest
        # AND rest must be a clique
        if is_clique[rest] and (rest & ~adj_mask[u]) == 0:
            is_clique[S] = True
            
        # Independent set check
        # u must NOT be connected to any vertex in rest
        # AND rest must be an independent set
        # non_adj_mask[u] is bits for v != u where (u,v) not in edges.
        # We can compute non_adj on the fly or precompute.
        # ~adj_mask[u] has 1s for non-neighbors, but also 1 for u itself and higher bits.
        # We only care about bits in 'rest'.
        # So (rest & adj_mask[u]) == 0 ensures u is not connected to anyone in rest.
        if is_indep[rest] and (rest & adj_mask[u]) == 0:
            is_indep[S] = True
            
        valid[S] = is_clique[S] or is_indep[S]
        
    # DP for minimum partition
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    # Iterate masks
    # To optimize, we can iterate through valid submasks only?
    # Standard submask iteration is fine for N=14.
    
    for mask in range(1, 1 << n):
        # Iterate through all submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct
    result = []
    mask = full_mask
    while mask:
        s = parent[mask]
        # Extract vertices from s
        part = []
        for i in range(n):
            if (s >> i) & 1:
                part.append(i)
        result.append(part)
        mask ^= s
        
    return result
