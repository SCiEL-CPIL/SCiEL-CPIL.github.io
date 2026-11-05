def construct():
    from math import isqrt
    import numpy as np

    # Generate all primes <= 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
    
    n = len(primes)
    if n < 2:
        return primes

    max_gap = 10000
    
    # dp[i][g] = max length of subseq ending at i with last gap exactly g
    # parent[i][g] = (j, prev_gap)
    # Using 0 for "no sequence" or length 0?
    # Base case: length 1.
    # If we use 0 for empty, then length 2 will be 0+1=1? No.
    # Let's stick to the logic:
    # pref_val[j][g] is max length ending at j with gap <= g.
    # For base case (length 1), pref_val is 1.
    
    # We can avoid storing dp as a separate array if we compute pref directly?
    # No, we need dp to build pref.
    
    dp = np.zeros((n, max_gap + 1), dtype=np.int32)
    parent = np.zeros((n, max_gap + 1, 2), dtype=np.int32)
    
    pref_val = np.zeros((n, max_gap + 1), dtype=np.int32)
    pref_idx = np.zeros((n, max_gap + 1), dtype=np.int32)
    
    # Initialize i=0
    # dp[0] is all 0.
    # pref_val[0] should be 1 for all g (base case length 1)
    pref_val[0] = 1
    pref_idx[0] = 0
    
    for i in range(1, n):
        pi = primes[i]
        # Update dp[i]
        for j in range(i):
            g = pi - primes[j]
            if g > max_gap:
                continue
            
            best_len = pref_val[j][g]
            prev_gap = pref_idx[j][g]
            
            candidate_len = best_len + 1
            
            if candidate_len > dp[i][g]:
                dp[i][g] = candidate_len
                parent[i][g, 0] = j
                parent[i][g, 1] = prev_gap
        
        # Build pref_val[i]
        # Start with base 1
        pref_val[i] = 1
        pref_idx[i] = 0
        
        # Update with dp[i]
        # dp[i] is an array. We can vectorize the update?
        # We want pref_val[i][g] = max(1, dp[i][g])
        # But we also need to propagate prefix max.
        
        # First, set pref_val[i] to max(1, dp[i])
        np.maximum(dp[i], 1, out=pref_val[i])
        
        # Now prefix max
        # Loop is O(G)
        for g in range(1, max_gap + 1):
            if pref_val[i][g-1] > pref_val[i][g]:
                pref_val[i][g] = pref_val[i][g-1]
                pref_idx[i][g] = pref_idx[i][g-1]
            else:
                # If equal or greater, we keep current idx?
                # If dp[i][g] > pref_val[i][g-1], then pref_val[i][g] is dp[i][g] (or 1).
                # And pref_idx[i][g] should be g (if dp[i][g] > 1) or 0 (if 1).
                # Our initialization sets pref_idx to 0.
                # If dp[i][g] > 1, we should set pref_idx[i][g] = g.
                # But we didn't do that in the vectorized step.
                # The vectorized step `np.maximum` only updates values, not indices.
                # So we need to update indices where dp[i][g] > 1.
                pass
        
        # Corrected Index Update
        # We need to set pref_idx[i][g] = g wherever dp[i][g] > 1.
        # Because if dp[i][g] is the max, the gap is g.
        # If dp[i][g] <= 1, the gap is 0 (base).
        
        mask = dp[i] > 1
        pref_idx[i][mask] = np.where(mask)[0]
        
        # Now do prefix max with indices
        for g in range(1, max_gap + 1):
            if pref_val[i][g-1] > pref_val[i][g]:
                pref_val[i][g] = pref_val[i][g-1]
                pref_idx[i][g] = pref_idx[i][g-1]
            # If <=, we keep current pref_val and pref_idx.
            # If pref_val[i][g] == pref_val[i][g-1], and pref_idx[i][g] is 0 (base) 
            # but pref_val[i][g-1] came from a real gap, we might want to prefer the real gap?
            # No, base gap 0 is valid for any g >= 0.
            # But if we have a real gap with same length, either is fine.
            
    # Find global max
    max_len = 0
    best_i = -1
    best_g = -1
    
    # We can just look at pref_val? No, pref_val is prefix max.
    # The max length ending at i is max(dp[i]).
    # Or max(pref_val[i]) is the same.
    # But we need the specific gap to backtrack.
    # The best gap for i is the one that gives max(dp[i]).
    
    for i in range(n):
        # Find max in dp[i]
        # If dp[i] is all 0, max is 0.
        # But we want length >= 2?
        # The problem implies a sequence.
        # If max_len is 1, we return a single prime?
        # "Return increasing prime integers... Maximize length".
        # Usually length >= 1.
        
        current_max = np.max(dp[i])
        if current_max > max_len:
            max_len = current_max
            best_i = i
            # Find the gap
            # If multiple, pick any.
            best_g = np.argmax(dp[i])
            
    if best_i == -1 or max_len <= 1:
        return primes[:1] # or primes[:2] if n>=2?
        # If no sequence of length 2, return length 1.
        # But primes[:1] is [2].
        
    # Reconstruct
    seq = []
    cur_i = best_i
    cur_g = best_g
    
    while cur_i != -1:
        seq.append(primes[cur_i])
        if cur_i == 0:
            break
        j, prev_g = parent[cur_i, cur_g]
        cur_i = j
        cur_g = prev_g
        
    seq.reverse()
    return seq
