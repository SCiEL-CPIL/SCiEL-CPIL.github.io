import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85
    
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
    # We use a 2D list of lists or numpy array.
    # N is approx 1229. N^2 is ~1.5M. Numpy is good.
    dp = np.full((N, N), 2, dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    best_len = 2
    best_end = (0, 1) if N > 1 else (0, 0)
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Loop i from 0 to N-1
    for i in range(N):
        # We compute dp[i][j] for j > i
        # k_limit is the smallest index k < i such that p_k >= 2*p_i - p_j
        # As j increases, 2*p_i - p_j decreases, so k_limit can decrease
        k_limit = i # Start with no valid k (range [i, i-1] is empty)
        max_val = 0
        best_k = -1
        
        # Precompute 2*p_i
        two_pi = 2 * primes[i]
        
        for j in range(i + 1, N):
            # Update k_limit
            # Condition: p_k >= two_pi - p_j
            # We want smallest k.
            # If p[k_limit - 1] >= two_pi - p_j, then k_limit - 1 is valid and smaller.
            # We can check this in a while loop.
            # Since j increases, two_pi - p_j decreases.
            # So the condition becomes easier to satisfy for smaller k.
            
            # Optimization: check if we can decrease k_limit
            # p[k_limit - 1] >= two_pi - p[j]
            # p[j] is primes_arr[j]
            # We can use a while loop.
            # Note: k_limit must be > 0 to access k_limit - 1
            
            # To avoid repeated array lookups, we can just check.
            # The while loop runs at most i times for this i.
            
            # However, we can also just binary search for the new k_limit if it's faster?
            # No, the amortized O(1) update is better.
            
            # But wait, if k_limit is already 0, we can't go lower.
            
            # Let's implement the while loop carefully.
            # We need to ensure we don't access out of bounds.
            
            # Actually, we can just check the next candidate.
            # But multiple k's might become valid at once?
            # No, k_limit is an index. We decrease it one by one.
            
            # Is it possible that k_limit - 1 is valid, but k_limit - 2 is also valid?
            # Yes. The while loop handles that.
            
            # One optimization: if k_limit == 0, break the while loop.
            
            # We can also pre-check: if k_limit > 0 and primes[k_limit - 1] >= two_pi - primes[j]:
            #   then enter while.
            
            while k_limit > 0 and primes[k_limit - 1] >= two_pi - primes[j]:
                k_limit -= 1
                # Add k_limit to the max calculation
                val = dp[k_limit, i]
                if val > max_val:
                    max_val = val
                    best_k = k_limit
            
            if best_k != -1:
                new_len = max_val + 1
                dp[i, j] = new_len
                parent[i, j] = best_k
                if new_len > best_len:
                    best_len = new_len
                    best_end = (i, j)
            else:
                dp[i, j] = 2
                parent[i, j] = -1
            
        # Check time periodically
        if i % 100 == 0 and time.time() > deadline:
            break

    # Reconstruct the sequence
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    while parent[ci, cj] != -1:
        ck = parent[ci, cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
        
    result.reverse()
    return result
