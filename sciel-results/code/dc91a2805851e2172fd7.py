import time
import math
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin

    N_LIMIT = 10000
    
    # Sieve of Eratosthenes using NumPy for speed
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
            
    primes = np.array([i for i, val in enumerate(is_prime) if val])
    n = len(primes)
    
    if n < 2:
        return primes.tolist()
        
    # dp[i][j] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i
    # We store prev[i][j] to reconstruct the path
    # Using lists of lists for flexibility and speed in Python loops
    # dp[i][j] is accessed as dp[i][j] in current code?
    # Current code: dp[i][j] where i is current, j is previous.
    # Loop: for j in range(n): ... for i in range(j+1, n): ...
    # So we fill column j of dp.
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # Precompute log2 for binary search? Or just use bisect.
    import bisect
    
    for j in range(n):
        if j == 0:
            # dp[i][0] = 2 for all i > 0
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            # For fixed j, we need to query max of dp[j][k] for k in [k_min, j-1]
            # Precompute suffix max and suffix argmax for the row dp[j][0...j-1]
            # suf_val[k] = max(dp[j][k], suf_val[k+1])
            # suf_idx[k] = argmax
            
            suf_val = [0] * (j + 1)
            suf_idx = [-1] * (j + 1)
            
            # Initialize the last element (k = j-1)
            suf_val[j-1] = dp[j][j-1]
            suf_idx[j-1] = j-1
            
            # Fill backwards
            for k in range(j-2, -1, -1):
                if dp[j][k] > suf_val[k+1]:
                    suf_val[k] = dp[j][k]
                    suf_idx[k] = k
                else:
                    # Tie-breaking: keep the one with smaller index? 
                    # The problem doesn't specify which valid sequence to return, just maximize length.
                    # However, for reconstruction, any valid path works.
                    # Let's keep the existing suf_idx[k+1] if equal, or k?
                    # Usually, smaller index might be "better" for future steps? 
                    # Actually, the value is the same. The index determines the previous element.
                    # Let's just pick k if equal to prefer smaller primes? 
                    # Or just keep the first one found (which would be the smaller index if we iterate backwards? No, backwards we see larger indices first).
                    # Let's stick to > for replacement, so if equal, we keep the one from k+1 (larger index).
                    # This is fine.
                    suf_val[k] = suf_val[k+1]
                    suf_idx[k] = suf_idx[k+1]
            
            # Now process i > j
            for i in range(j + 1, n):
                gap = primes[i] - primes[j]
                target = primes[j] - gap
                
                # Find smallest k in [0, j-1] such that primes[k] >= target
                # primes is sorted.
                # bisect_left returns the index of the first element >= target
                k_min = bisect.bisect_left(primes, target, 0, j)
                
                if k_min < j:
                    max_val = suf_val[k_min]
                    best_k = suf_idx[k_min]
                    dp[i][j] = 1 + max_val
                    prev[i][j] = best_k
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct sequence
    if best_i == -1:
        return [primes[0]]
        
    seq = [int(primes[best_i]), int(primes[best_j])]
    cur_i = best_i
    cur_j = best_j
    while prev[cur_i][cur_j] != -1:
        cur_k = prev[cur_i][cur_j]
        seq.append(int(primes[cur_k]))
        cur_i, cur_j = cur_j, cur_k
        
    seq.reverse()
    return seq
