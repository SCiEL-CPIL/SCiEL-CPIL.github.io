"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from bisect import bisect_left

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
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # Use list of lists for dp and prev
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # suf_max_val[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j])
    # suf_max_arg[j][k] = index of the argmax in that range
    suf_max_val = [None] * N
    suf_max_arg = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    # Local references for speed
    p = primes
    bl = bisect_left
    
    for j in range(1, N):
        if time.time() > deadline:
            break
        
        pj = p[j]
        dp_j_col = dp  # We'll access dp[i][j] directly
        
        for i in range(0, j):
            gap = pj - p[i]
            threshold = p[i] - gap
            
            # Find L: smallest index in [0, i-1] such that primes[L] >= threshold
            L = bl(p, threshold, 0, i)
            
            if L < i:
                val = suf_max_val[i][L]
                if val > 0:
                    dp[i][j] = val + 1
                    prev[i][j] = suf_max_arg[i][L]
                else:
                    dp[i][j] = 2
            else:
                dp[i][j] = 2
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # Compute suffix max value and argmax for column j
        sm_val = [0] * j
        sm_arg = [-1] * j
        if j > 0:
            sm_val[j-1] = dp[j-1][j]
            sm_arg[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[k][j] > sm_val[k+1]:
                    sm_val[k] = dp[k][j]
                    sm_arg[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_arg[k] = sm_arg[k+1]
        suf_max_val[j] = sm_val
        suf_max_arg[j] = sm_arg
    
    # Reconstruct sequence
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
