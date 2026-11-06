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
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    dp = np.full((N, N), 2, dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    best_len = 2
    best_end = (0, 1)
    
    # For each i, we'll maintain a data structure to query max dp[k][i] for k in [lo, i-1]
    # where lo is determined by the threshold
    
    # Precompute for each i, the array of dp[k][i] for k < i
    # and build a segment tree or sparse table for range max queries
    
    # Since N ~ 1229, N^2 ~ 1.5M, we can use a simpler approach:
    # For each i, precompute suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # and suffix_parent[i][k] = argmax
    
    suffix_max = [None] * N
    suffix_parent = [None] * N
    
    # Process j from 1 to N-1
    for j in range(1, N):
        # Before computing dp[i][j] for all i < j, we need suffix structures for i = j-1
        # But actually, dp[i][j] depends on dp[k][i] for k < i, which were computed in previous j iterations
        # So we need suffix_max[i] to be up-to-date for all i < j
        
        # Compute suffix_max and suffix_parent for i = j-1 (and potentially others if not yet computed)
        # Actually, we need suffix_max[i] for all i that will be used, which is all i < j
        # But dp[k][i] is only finalized when we've processed all j' > i
        # Wait, no: dp[k][i] is computed when j=i, so by the time we're at j, all dp[k][i] for i < j are finalized
        
        # Let's compute suffix structures incrementally
        # For i = j-1, we can compute it now since all dp[k][j-1] for k < j-1 are already set
        i_val = j - 1
        if i_val > 0:
            col = dp[:i_val, i_val].copy()
            # Build suffix max
            sm = np.empty(i_val, dtype=np.int32)
            sm[i_val - 1] = col[i_val - 1]
            sp = np.empty(i_val, dtype=np.int32)
            sp[i_val - 1] = i_val - 1
            for k in range(i_val - 2, -1, -1):
                if col[k] >= sm[k + 1]:
                    sm[k] = col[k]
                    sp[k] = k
                else:
                    sm[k] = sm[k + 1]
                    sp[k] = sp[k + 1]
            suffix_max[i_val] = sm
            suffix_parent[i_val] = sp
        else:
            suffix_max[i_val] = np.array([], dtype=np.int32)
            suffix_parent[i_val] = np.array([], dtype=np.int32)
        
        # Now compute dp[i][j] for all i < j
        pj = primes[j]
        for i in range(j):
            # Find the smallest k such that primes[k] >= 2*primes[i] - pj
            threshold = 2 * primes[i] - pj
            # k must satisfy: k < i and primes[k] >= threshold
            # Also k >= 0
            # Find the leftmost index in primes_arr where value >= threshold
            km = int(np.searchsorted(primes_arr, threshold, side='left'))
            # k must be < i, so effective lower bound is max(km, 0), upper bound is i-1
            # If km >= i, no valid k exists
            if km < i:
                # Query suffix_max[i][km]
                sm_i = suffix_max[i]
                if sm_i is not None and km < len(sm_i):
                    val = 1 + sm_i[km]
                    if val > dp[i, j]:
                        dp[i, j] = val
                        parent[i, j] = suffix_parent[i][km]
                        if val > best_len:
                            best_len = val
                            best_end = (i, j)
        
        # Check time periodically
        if j % 20 == 0 and time.time() > deadline:
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
