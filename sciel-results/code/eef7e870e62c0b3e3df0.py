import time
from math import isqrt
from bisect import bisect_left
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 10000
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = bytearray(len(range(i*i, N+1, i)))
    
    primes = [i for i in range(2, N+1) if is_prime[i]]
    K = len(primes)
    
    if K < 2:
        return primes

    primes_arr = np.array(primes, dtype=np.int32)
    
    # dp[j][k] = length of longest valid subsequence ending with primes[j], primes[k]
    # where j < k and primes[j], primes[k] are the last two elements
    # prev[j][k] = index i such that the subsequence ending with primes[i], primes[j] was extended to primes[k]
    
    dp = np.full((K, K), 2, dtype=np.int16)
    prev = np.full((K, K), -1, dtype=np.int32)
    
    max_len = 2
    best_end_j = 0
    best_end_k = 1

    # For each j from 1 to K-1:
    #   For each k > j:
    #     We need i < j such that primes[j] - primes[i] <= primes[k] - primes[j]
    #     i.e., primes[i] >= 2*primes[j] - primes[k]
    #     dp[j][k] = max(2, max(dp[i][j] + 1 for valid i))
    
    for j in range(1, K):
        if time.time() > deadline:
            break
        
        # For each k > j, compute the minimum valid i
        # L = 2*primes[j] - primes[k]
        # i_min = first index where primes[i] >= L, among i in [0, j)
        
        # We want to compute for all k > j simultaneously
        # For each k, i_min = bisect_left(primes, L, 0, j)
        # Then dp[j][k] = max(2, max(dp[i][j] for i in [i_min, j-1]) + 1) if i_min < j
        
        # Build suffix maximum of dp[0..j-1][j]
        # SufMax[i] = max(dp[t][j] for t in [i, j-1])
        # ArgSufMax[i] = argmax t
        
        col = dp[:j, j]  # shape (j,)
        
        SufMax = np.zeros(j, dtype=np.int16)
        ArgSufMax = np.zeros(j, dtype=np.int32)
        
        SufMax[j-1] = col[j-1]
        ArgSufMax[j-1] = j-1
        
        for i in range(j-2, -1, -1):
            if col[i] > SufMax[i+1]:
                SufMax[i] = col[i]
                ArgSufMax[i] = i
            else:
                SufMax[i] = SufMax[i+1]
                ArgSufMax[i] = ArgSufMax[i+1]
        
        # For k in range(j+1, K):
        #   L = 2*primes[j] - primes[k]
        #   i_min = bisect_left(primes, L, 0, j)
        #   if i_min < j: new_len = SufMax[i_min] + 1
        
        k_indices = np.arange(j+1, K)
        L_vals = 2 * primes_arr[j] - primes_arr[k_indices]  # shape (K-j-1,)
        
        # Find i_min for each L_val using searchsorted
        i_mins = np.searchsorted(primes_arr[:j], L_vals, side='left')
        
        # Valid where i_mins < j
        valid_mask = i_mins < j
        
        if np.any(valid_mask):
            valid_k_indices = k_indices[valid_mask]
            valid_i_mins = i_mins[valid_mask]
            new_lens = SufMax[valid_i_mins] + 1
            
            # Update dp[j][k] where new_len > dp[j][k]
            current_vals = dp[j, valid_k_indices]
            improve_mask = new_lens > current_vals
            
            if np.any(improve_mask):
                improve_k = valid_k_indices[improve_mask]
                improve_i = valid_i_mins[improve_mask]
                improve_len = new_lens[improve_mask]
                
                dp[j, improve_k] = improve_len
                prev[j, improve_k] = ArgSufMax[improve_i]
                
                # Track max
                max_new = int(np.max(improve_len))
                if max_new > max_len:
                    idx = int(np.argmax(improve_len))
                    max_len = max_new
                    best_end_j = j
                    best_end_k = int(improve_k[idx])

    # Reconstruct sequence
    if max_len == 2:
        return primes[:2]
        
    seq = [primes[best_end_k], primes[best_end_j]]
    cur_j = best_end_j
    cur_k = best_end_k
    
    while prev[cur_j][cur_k] != -1:
        prev_i = int(prev[cur_j][cur_k])
        seq.append(primes[prev_i])
        cur_k = cur_j
        cur_j = prev_i
        
    seq.reverse()
    return seq
