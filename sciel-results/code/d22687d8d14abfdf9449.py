import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # dp_cols[j][i] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # We use a list of lists, where the outer list is indexed by j (the second prime in the pair).
    dp_cols = [[0] * N for _ in range(N)]
    prev_cols = [[-1] * N for _ in range(N)]
    
    # suf_max_val[i] and suf_max_idx[i] for column i
    # suf_max_val[i][k] = max(dp_cols[i][m]) for m in [k, i-1]
    # suf_max_idx[i][k] = argmax m in [k, i-1]
    suf_max_val = [None] * N
    suf_max_idx = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    bisect_left = bisect.bisect_left
    
    for j in range(1, N):
        # Check time
        if time.time() > deadline:
            break
            
        p_j = primes[j]
        dp_col_j = dp_cols[j]
        prev_col_j = prev_cols[j]
        
        # Compute dp[i][j] for all i < j
        for i in range(0, j):
            p_i = primes[i]
            gap = p_j - p_i
            threshold = p_i - gap
            
            # Find smallest k < i such that primes[k] >= threshold
            L = bisect_left(primes, threshold, 0, i)
            
            if L < i:
                # We have valid k in [L, i-1]
                # Get max from suf_max_val[i]
                val = suf_max_val[i][L]
                idx = suf_max_idx[i][L]
                
                val_plus_1 = val + 1
                dp_col_j[i] = val_plus_1
                prev_col_j[i] = idx
                
                if val_plus_1 > max_len:
                    max_len = val_plus_1
                    best_i = i
                    best_j = j
            else:
                dp_col_j[i] = 2
                prev_col_j[i] = -1
                
                if 2 > max_len:
                    max_len = 2
                    best_i = i
                    best_j = j
        
        # Compute suf_max for column j
        # Column j has entries dp_cols[j][0], ..., dp_cols[j][j-1]
        # We need suf_max_val[j] of size j
        # suf_max_val[j][k] = max(dp_cols[j][m]) for m in [k, j-1]
        if j > 0:
            sm_val = [0] * j
            sm_idx = [0] * j
            
            # Initialize with last element m = j-1
            sm_val[j-1] = dp_col_j[j-1]
            sm_idx[j-1] = j-1
            
            for k in range(j-2, -1, -1):
                if dp_col_j[k] > sm_val[k+1]:
                    sm_val[k] = dp_col_j[k]
                    sm_idx[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_idx[k] = sm_idx[k+1]
            
            suf_max_val[j] = sm_val
            suf_max_idx[j] = sm_idx
    
    # Reconstruct sequence
    # Start from best_j, best_i
    # Sequence ends with primes[best_i], primes[best_j]
    # prev[best_i][best_j] is stored in prev_cols[best_j][best_i]
    
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    
    # Safety check to prevent infinite loops or errors
    steps = 0
    while curr_i != -1 and steps < N:
        seq.append(primes[curr_i])
        prev_i = prev_cols[curr_j][curr_i]
        curr_j = curr_i
        curr_i = prev_i
        steps += 1
    
    seq.reverse()
    
    return seq
