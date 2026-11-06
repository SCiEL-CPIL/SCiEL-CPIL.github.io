import time
import math
from bisect import bisect_left

def construct():
    deadline = time.time() + 78  # 78s deadline, 12s safety margin
    
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
    
    if N <= 1:
        return primes
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    # We store dp as a 2D list, dp[i][j] for i < j
    dp = [[2] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # Precompute suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i]) for all i, k < i
    # suffix_max_parent[i][k] = the index that achieves the max
    suffix_max = [[0] * N for _ in range(N)]
    suffix_max_parent = [[-1] * N for _ in range(N)]
    
    best_len = 2
    best_end = (0, 1)
    
    two_primes = [2 * p for p in primes]
    
    for j in range(1, N):
        # Now compute dp[i][j] for i in range(j)
        for i in range(j):
            if time.time() > deadline:
                break
            threshold = two_primes[i] - primes[j]
            # Find smallest k such that primes[k] >= threshold in primes[0:i]
            k_min = bisect_left(primes, threshold, 0, i)
            
            if k_min < i:
                val = 1 + suffix_max[i][k_min]
                if val > dp[i][j]:
                    dp[i][j] = val
                    parent[i][j] = suffix_max_parent[i][k_min]
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_end = (i, j)
        
        # After finishing all i for this j, update suffix_max for row j
        # suffix_max[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j]) for k in 0..j-1
        for k in range(j - 1, -1, -1):
            if k == j - 1:
                suffix_max[j][k] = dp[k][j]
                suffix_max_parent[j][k] = k
            else:
                if dp[k][j] > suffix_max[j][k + 1]:
                    suffix_max[j][k] = dp[k][j]
                    suffix_max_parent[j][k] = k
                else:
                    suffix_max[j][k] = suffix_max[j][k + 1]
                    suffix_max_parent[j][k] = suffix_max_parent[j][k + 1]
        
        if time.time() > deadline:
            break
    
    # Reconstruct the sequence via backtracking
    if best_len <= 1:
        return primes[:best_len]
    
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    
    while parent[ci][cj] != -1:
        ck = parent[ci][cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
    
    result.reverse()
    return result
