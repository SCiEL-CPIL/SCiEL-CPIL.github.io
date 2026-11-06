import time
import math
import bisect

def construct():
    deadline = time.time() + 50  # 50s deadline, 10s safety margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
        
    # dp[i][j] = max length of valid subseq ending with primes[j], primes[i] (j < i)
    # prev_idx[i][j] = index k such that primes[k] precedes primes[j] in the optimal subseq for dp[i][j]
    dp = [[0] * N for _ in range(N)]
    prev_idx = [[-1] * N for _ in range(N)]
    
    max_len = 2
    best_i, best_j = 0, 1
    
    # Iterate j from 0 to N-1
    # For each j, we have computed dp[j][k] for all k < j
    for j in range(N):
        # Precompute suffix maximums for dp[j][0..j-1]
        # max_from_k[k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
        # arg_max_k[k] = index t in [k, j-1] that achieves this max
        if j > 0:
            max_from_k = [0] * j
            arg_max_k = [-1] * j
            curr_max = 0
            curr_arg = -1
            for k in range(j - 1, -1, -1):
                if dp[j][k] > curr_max:
                    curr_max = dp[j][k]
                    curr_arg = k
                max_from_k[k] = curr_max
                arg_max_k[k] = curr_arg
        
        # Iterate i from j+1 to N-1
        for i in range(j + 1, N):
            # Condition: primes[i] - primes[j] >= primes[j] - primes[k]
            # primes[k] >= 2 * primes[j] - primes[i]
            target = 2 * primes[j] - primes[i]
            k_min = bisect.bisect_left(primes, target)
            
            if k_min < j:
                best_len = max_from_k[k_min]
                best_k = arg_max_k[k_min]
                dp[i][j] = 1 + best_len
                prev_idx[i][j] = best_k
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i, best_j = i, j
            else:
                dp[i][j] = 2
                prev_idx[i][j] = -1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i, best_j = i, j
                    
    # Reconstruct sequence
    indices = [best_i, best_j]
    i, j = best_i, best_j
    while prev_idx[i][j] != -1:
        k = prev_idx[i][j]
        indices.append(k)
        i, j = j, k
        
    indices.reverse()
    return [primes[idx] for idx in indices]
