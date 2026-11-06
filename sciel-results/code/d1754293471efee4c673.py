import bisect
from math import isqrt

def construct():
    # Generate primes up to 10000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    P = get_primes(10000)
    N = len(P)
    
    # dp[i][j] = length of longest valid subsequence ending with P[i], P[j] (i < j)
    # We only care about i < j.
    # To save memory and time, we can use a 2D list or list of lists.
    # N is ~1229. N^2 is ~1.5M. Integers are small.
    # dp[j][i] where j < i.
    
    dp = [[0] * N for _ in range(N)]
    par = [[-1] * N for _ in range(N)]
    
    max_len = 0
    best_j = -1
    best_i = -1
    
    # Iterate j from 0 to N-1
    for j in range(N):
        # Prepare suffix max for column j (rows 0 to j-1)
        # suf[k] = (max_val, index) for k in range(k, j)
        # We only need this for k < j.
        if j == 0:
            continue
            
        suf_val = [0] * j
        suf_idx = [-1] * j
        
        # Initialize last element
        suf_val[j-1] = dp[j-1][j]
        suf_idx[j-1] = j-1
        
        for k in range(j-2, -1, -1):
            if dp[k][j] > suf_val[k+1]:
                suf_val[k] = dp[k][j]
                suf_idx[k] = k
            else:
                suf_val[k] = suf_val[k+1]
                suf_idx[k] = suf_idx[k+1]
                
        # Iterate i from j+1 to N-1
        for i in range(j+1, N):
            # Condition: P[k] >= 2*P[j] - P[i]
            X = 2 * P[j] - P[i]
            
            # Find smallest k in [0, j) such that P[k] >= X
            # bisect_left returns first index where P[idx] >= X
            k_min = bisect.bisect_left(P, X, 0, j)
            
            if k_min < j:
                val = suf_val[k_min]
                k_best = suf_idx[k_min]
                dp[j][i] = val + 1
                par[j][i] = k_best
            else:
                # No valid k
                dp[j][i] = 2
                par[j][i] = -1
                
            if dp[j][i] > max_len:
                max_len = dp[j][i]
                best_j = j
                best_i = i
                
    # Reconstruct sequence
    if max_len == 0:
        return []
        
    seq = [P[best_i], P[best_j]]
    j_cur = best_j
    i_cur = best_i
    
    while par[j_cur][i_cur] != -1:
        k = par[j_cur][i_cur]
        seq.append(P[k])
        i_cur = j_cur
        j_cur = k
        
    seq.reverse()
    return seq
