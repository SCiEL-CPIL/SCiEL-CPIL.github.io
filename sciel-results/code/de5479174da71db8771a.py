import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Generate primes up to 10000
    primes = []
    for p in range(2, 10001):
        if is_prime(p):
            primes.append(p)
            
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i][j] = length of longest valid sequence ending in primes[j], primes[i]
    # prev_idx[i][j] = index k such that primes[k], primes[j], primes[i] is the optimal extension
    # if prev_idx is -1, the sequence starts at primes[j], primes[i] (length 2)
    
    dp = [[0] * n for _ in range(n)]
    prev_idx = [[-1] * n for _ in range(n)]
    
    # suffix_max_val[j][k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
    # suffix_max_idx[j][k] = index of the max in dp[j][k..j-1]
    suffix_max_val = [None] * n
    suffix_max_idx = [None] * n
    
    best_len = 1
    best_i = 0
    best_j = -1
    
    # Initialize for j=0 (no k < 0)
    # suffix_max_val[0] is empty
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        for j in range(i):
            gap = primes[i] - primes[j]
            threshold = primes[j] - gap
            
            # Find k_min such that primes[k_min] >= threshold
            # Search in primes[0..j-1]
            k_min = bisect.bisect_left(primes, threshold, 0, j)
            
            max_prev_val = 0
            best_k = -1
            
            if k_min < j:
                # We have valid k in [k_min, j-1]
                max_prev_val = suffix_max_val[j][k_min]
                best_k = suffix_max_idx[j][k_min]
                
            # dp[i][j] = 1 + max(1, max_prev_val)
            if max_prev_val == 0:
                dp[i][j] = 2
                prev_idx[i][j] = -1
            else:
                dp[i][j] = max_prev_val + 1
                prev_idx[i][j] = best_k
                
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
                
        # Compute suffix max for row i
        if i > 0:
            smax_val = [0] * i
            smax_idx = [0] * i
            smax_val[i-1] = dp[i][i-1]
            smax_idx[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[i][k] >= smax_val[k+1]:
                    smax_val[k] = dp[i][k]
                    smax_idx[k] = k
                else:
                    smax_val[k] = smax_val[k+1]
                    smax_idx[k] = smax_idx[k+1]
            suffix_max_val[i] = smax_val
            suffix_max_idx[i] = smax_idx

    # Reconstruct path
    if best_len == 1:
        return [primes[best_i]]
        
    path = []
    curr_i = best_i
    curr_j = best_j
    
    while curr_j != -1:
        path.append(primes[curr_i])
        path.append(primes[curr_j])
        curr_j, curr_i = prev_idx[curr_i][curr_j], curr_j
        
    path.reverse()
    return path

# EVOLVE-BLOCK-END
