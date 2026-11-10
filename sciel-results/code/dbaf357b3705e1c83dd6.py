import time
import numpy as np

def _is_prime_sieve(n):
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 50  # Safety margin
    
    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
        
    max_gap = N_LIMIT
    # dp[i][g] = max length ending at i with last gap g
    dp = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # PMax[i][g] = max_{h<=g} dp[i][h]
    PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # argmax_PMax[i][g] = argmax_{h<=g} dp[i][h]
    argmax_PMax = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # parent tracking
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    
    # primes array for fast access
    p_arr = np.array(primes, dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
            
        if j > 0:
            i_range = np.arange(j)
            gaps = p_arr[j] - p_arr[i_range]
            
            # Get previous max lengths
            # Note: PMax[i][g] is available for all i < j
            prev_lens = PMax[i_range, gaps]
            
            # Calculate candidates
            cands = np.where(prev_lens > 0, prev_lens + 1, 2)
            
            # Current dp values for these gaps
            current_dp = dp[j, gaps]
            
            # Find where candidate is better
            mask = cands > current_dp
            
            if np.any(mask):
                # Update dp
                dp[j, gaps[mask]] = cands[mask]
                
                # Update parents
                parent_i[j, gaps[mask]] = i_range[mask]
                parent_gap[j, gaps[mask]] = argmax_PMax[i_range[mask], gaps[mask]]
        
        # Compute PMax[j] and argmax_PMax[j]
        # PMax[j][g] = max(PMax[j][g-1], dp[j][g])
        # We can do this with np.maximum.accumulate
        PMax[j, :] = np.maximum.accumulate(dp[j, :])
        
        # Compute argmax_PMax[j] efficiently using numpy
        # We need argmax_PMax[j][g] = argmax_{h<=g} dp[j][h]
        # This can be computed by finding the last occurrence of the running maximum
        
        # Method: 
        # 1. Find the running maximum (already done in PMax)
        # 2. For each position g, find the last index h <= g where dp[j][h] == PMax[j][g]
        
        # Alternative efficient method:
        # Process from right to left, tracking the best candidate
        # But we need prefix argmax, so let's use a different approach
        
        # We can use the fact that if dp[j][g] > running_max, then argmax becomes g
        # Otherwise, it stays the same
        # This is exactly what the loop does, but let's try to vectorize it
        
        # Actually, the loop is O(max_gap) per j, and max_gap = 10000, n_primes = 1229
        # Total: 12.29M iterations, which should be fast enough in Python
        
        dp_j = dp[j]
        argmax_j = argmax_PMax[j]
        
        # Use a faster approach: find all positions where dp_j exceeds the previous running max
        # Then fill in the argmax array
        
        # Find indices where dp_j has a new maximum
        # running_max starts at 0
        # We want to find for each g, the largest h <= g such that dp_j[h] is the maximum in dp_j[0..g]
        
        # Simple approach: iterate and track
        running_max = 0
        running_arg = 0
        
        # Vectorized approach using cumulative operations
        # This is tricky, so let's stick with the loop but optimize it
        
        # Optimization: only update when dp_j[g] > running_max
        for g in range(1, max_gap + 1):
            if dp_j[g] > running_max:
                running_max = dp_j[g]
                running_arg = g
            argmax_j[g] = running_arg

    # Find global max
    max_len = 0
    best_j = -1
    best_gap = -1
    
    # Find the maximum in the entire dp array
    flat_max = np.max(dp)
    if flat_max > 0:
        max_len = int(flat_max)
        # Find the first occurrence of the maximum
        flat_idx = np.argmax(dp)
        best_j, best_gap = np.unravel_index(flat_idx, dp.shape)
        best_j = int(best_j)
        best_gap = int(best_gap)
    
    if max_len < 2:
        return [primes[0]]
        
    # Reconstruct
    result = []
    cur_j = best_j
    cur_gap = best_gap
    
    while cur_j >= 0:
        result.append(primes[cur_j])
        pi = parent_i[cur_j, cur_gap]
        pg = parent_gap[cur_j, cur_gap]
        
        if pi < 0:
            break
            
        cur_j = int(pi)
        cur_gap = int(pg)
        
        # If the length at this step was 2, the parent is the start
        # How do we know? 
        # If dp[cur_j][cur_gap] == 2, then it was formed from a single prime.
        # In that case, parent_gap would be argmax_PMax[pi][gap] which is 0 (since dp[pi] was 0).
        # So if cur_gap == 0, we append the last prime and stop.
        if cur_gap == 0:
            result.append(primes[cur_j])
            break
            
    result.reverse()
    
    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
             # Fallback: greedy approach
             values = []
             for p in primes:
                 if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                     values.append(p)
             return values
             
    return result
