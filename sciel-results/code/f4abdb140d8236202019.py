import time

P = {'n': 14, 'edges': ...}

def construct():
    n = P['n']
    # Build adj masks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Generate all valid candidates (cliques or independent sets)
    candidates = []
    for mask in range(1, 1 << n):
        # Check clique
        is_clique = True
        # Check independent
        is_indep = True
        
        # Iterate bits
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # For clique: all other bits in mask must be in adj_mask[v]
            # adj_mask[v] & mask should be mask without v
            if (adj_mask[v] & mask) != (mask ^ lsb):
                is_clique = False
                break
            # For indep: no other bits in mask should be in adj_mask[v]
            if (adj_mask[v] & mask) != 0:
                is_indep = False
                # We can't break both easily if we want to check both?
                # But we only need one to be true.
                # Actually, if is_clique is False, we still need to check is_indep?
                # Yes.
            m ^= lsb
            
        # Re-check is_indep if we didn't break? 
        # The loop above breaks if is_clique fails. 
        # If is_clique fails, we still need to know if it's indep.
        # So we should run two separate checks or one combined check that doesn't break early for the other.
        
        # Let's do it cleanly:
        is_clique = True
        is_indep = True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            others = mask ^ lsb
            if (adj_mask[v] & others) != others:
                is_clique = False
            if (adj_mask[v] & others) != 0:
                is_indep = False
            m ^= lsb
            
        if is_clique or is_indep:
            candidates.append(mask)
            
    # DP
    # dp[mask] = min parts to cover mask
    dp = [100] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n) # to store the candidate used
    
    for mask in range(1, 1 << n):
        best = 100
        best_cand = 0
        # Try all candidates
        for cand in candidates:
            if (cand & mask) == cand: # cand is subset of mask
                prev = mask ^ cand
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_cand = cand
        dp[mask] = best
        parent[mask] = best_cand
        
    # Reconstruct
    parts = []
    cur_mask = full_mask
    while cur_mask:
        cand = parent[cur_mask]
        if cand == 0:
            # Fallback
            break
        # Convert cand to list of vertices
        part = [i for i in range(n) if (cand >> i) & 1]
        parts.append(part)
        cur_mask ^= cand
        
    # Verify
    # ... (standard verification)
    
    return parts
