import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute validity of each mask
    # valid[mask] is True if mask is a clique or an independent set
    valid = [False] * (1 << n)
    
    # Helper to check clique
    def is_clique(mask):
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            others = mask ^ lsb
            if others & ~adj[u]:
                return False
        return True

    # Helper to check independent set
    def is_independent(mask):
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            others = mask ^ lsb
            if others & adj[u]:
                return False
        return True

    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            valid[mask] = True
    
    # DP: dp[mask] = min parts to cover mask
    # dp[0] = 0
    # dp[mask] = 1 + min(dp[mask ^ sub]) for sub in submasks of mask where valid[sub] is True
    
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # To reconstruct, we store the best submask chosen for each mask
    best_sub = [-1] * (1 << n)
    
    # We need to iterate masks in increasing order of popcount? 
    # Or just iterate 1 to full_mask?
    # dp[mask] depends on dp[mask ^ sub] where mask ^ sub is a strict submask of mask.
    # So if we iterate mask from 1 to full_mask, is it guaranteed that dp[mask ^ sub] is computed?
    # mask ^ sub is a submask of mask, so its value is smaller than mask?
    # Yes, if sub is non-empty, mask ^ sub < mask is not necessarily true in terms of integer value?
    # Wait. mask ^ sub is a submask. Any proper submask has a smaller integer value?
    # No. Example: mask = 1010 (10), sub = 0010 (2). mask ^ sub = 1000 (8). 8 < 10.
    # Example: mask = 1001 (9), sub = 0001 (1). mask ^ sub = 1000 (8). 8 < 9.
    # In general, if sub is a non-empty submask of mask, then mask ^ sub < mask?
    # Let mask = ...1...1...
    # Let sub = ...1...0... (1 at a position where mask has 1)
    # mask ^ sub will have 0 at that position and same bits elsewhere.
    # The highest bit where mask and sub differ is the highest set bit in sub.
    # At that bit, mask has 1, sub has 1, so mask ^ sub has 0.
    # All higher bits are the same as mask.
    # So mask ^ sub < mask.
    # Yes, iterating from 1 to full_mask works.
    
    for mask in range(1, 1 << n):
        best_val = INF
        best_s = -1
        sub = mask
        while sub:
            if valid[sub]:
                rest = mask ^ sub
                val = 1 + dp[rest]
                if val < best_val:
                    best_val = val
                    best_s = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        best_sub[mask] = best_s
        
    # Reconstruct
    partition = []
    current_mask = full_mask
    while current_mask:
        s = best_sub[current_mask]
        if s == -1:
            # Error
            break
        # Extract vertices
        part = []
        m = s
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            part.append(v)
            m -= lsb
        partition.append(part)
        current_mask ^= s
        
    return partition
