import math
import bisect
import time
import numpy as np

def construct():
    deadline = time.time() + 85  # Leave 5s safety margin for 90s limit
    limit = 10000
    
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[j][i] = max length of sequence ending with primes[j], primes[i] (j < i)
    # parent[j][i] = the m such that dp[m][j] was used to extend to dp[j][i]
    dp = np.zeros((N, N), dtype=np.int16)
    parent = np.full((N, N), -1, dtype=np.int16)
    
    # suf_max[j] = list of (max_val, arg_index) for suffix max of dp[m][j]
    # suf_max[j][k] = (max of dp[m][j] for m in [k, j-1], argmax index)
    suf_max = [None] * N
    
    max_len = 2
    best_i = 1
    best_j = 0
    
    for i in range(N):
        if time.time() > deadline:
            break
        for j in range(i):
            target = 2 * primes[j] - primes[i]
            # Find smallest index L such that primes[L] >= target
            L = bisect.bisect_left(primes, target)
            
            best_val = 0
            best_m = -1
            
            if L < j and suf_max[j] is not None:
                best_val, best_m = suf_max[j][L]
            
            if best_val > 0:
                dp[j, i] = best_val + 1
                parent[j, i] = best_m
            else:
                dp[j, i] = 2
                parent[j, i] = -1
                
            val = int(dp[j, i])
            if val > max_len:
                max_len = val
                best_i = i
                best_j = j
        
        # Build suf_max[i] for use in future iterations
        if i > 0:
            sm = [None] * i
            max_v = -1
            max_idx = -1
            for m in range(i - 1, -1, -1):
                v = int(dp[m, i])
                if v > max_v:
                    max_v = v
                    max_idx = m
                sm[m] = (max_v, max_idx)
            suf_max[i] = sm
    
    # Backtrack
    result = []
    cur_i = best_i
    cur_j = best_j
    result.append(primes[cur_i])
    result.append(primes[cur_j])
    
    while parent[cur_j, cur_i] != -1:
        prev_m = int(parent[cur_j, cur_i])
        result.append(primes[prev_m])
        cur_i = cur_j
        cur_j = prev_m
        
    result.reverse()
    return result
