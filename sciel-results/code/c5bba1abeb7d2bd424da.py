import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # DP with NumPy
    # dp[s] = min number of divisors to reach sum s
    # Using a large number instead of inf for integer arithmetic
    INF = 10000
    dp = np.full(target + 1, INF, dtype=np.int32)
    dp[0] = 0
    
    prev_sum = np.full(target + 1, -1, dtype=np.int32)
    prev_div_idx = np.full(target + 1, -1, dtype=np.int32)
    
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        
        # We need to update dp[s] for s from target down to d
        # dp[s] = min(dp[s], dp[s-d] + 1)
        # To do this in parallel, we need the old values of dp[s-d]
        
        # Slice of dp for s-d (indices 0 to target-d)
        # Slice of dp for s (indices d to target)
        
        # Create a copy of the relevant part of dp to avoid race conditions
        # Actually, we can just copy the whole dp or the relevant slice
        # old_dp_slice = dp[:target+1-d].copy()
        
        # Let's try:
        # new_candidates = dp[:target+1-d] + 1
        # updates = new_candidates < dp[d:]
        
        # If we use dp[:target+1-d] directly, it's a view.
        # If we assign to dp[d:], and d > 0, do they overlap?
        # Yes, if d < target/2.
        # So we MUST copy.
        
        old_vals = dp[:target+1-d].copy()
        
        new_vals = old_vals + 1
        current_vals = dp[d:]
        
        mask = new_vals < current_vals
        
        if np.any(mask):
            # Update dp
            dp[d:][mask] = new_vals[mask]
            
            # Update prev_sum and prev_div_idx
            # s = d + index_in_slice
            # indices in the slice corresponding to mask
            indices = np.where(mask)[0]
            s_values = d + indices
            
            prev_sum[s_values] = s_values - d
            prev_div_idx[s_values] = idx

    if dp[target] == INF:
        # Fallback
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        return chosen

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        idx = prev_div_idx[s]
        if idx == -1:
            break
        chosen.append(divisors[idx])
        s = prev_sum[s]
        
    return chosen
