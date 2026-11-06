"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Sieve for primes <= 10000
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    n = len(primes)  # 1229
    
    if n < 2:
        return primes.tolist()
    
    NEG_INF = -10**9
    
    # dp[i][j] = length of longest valid subseq ending at primes[i] with prev = primes[j]
    dp = np.full((n, n), NEG_INF, dtype=np.int32)
    prev_idx = np.full((n, n), -1, dtype=np.int32)
    
    best_len = 1
    best_i = 0
    best_j = -1
    
    for j in range(n):
        if j == 0:
            for i in range(1, n):
                dp[i, 0] = 2
                prev_idx[i, 0] = -1
                if 2 > best_len:
                    best_len = 2
                    best_i = i
                    best_j = 0
            continue
        
        # Build suffix max and argmax for dp[j][0..j-1]
        suf = np.empty(j, dtype=np.int32)
        suf_arg = np.empty(j, dtype=np.int32)
        suf[j-1] = dp[j, j-1]
        suf_arg[j-1] = j-1
        for m in range(j-2, -1, -1):
            if dp[j, m] >= suf[m+1]:
                suf[m] = dp[j, m]
                suf_arg[m] = m
            else:
                suf[m] = suf[m+1]
                suf_arg[m] = suf_arg[m+1]
        
        for i in range(j+1, n):
            gap = primes[i] - primes[j]
            threshold = 2 * primes[j] - primes[i]
            # Find smallest m where primes[m] >= threshold, within [0, j)
            m = np.searchsorted(primes, threshold, side='left', sorter=None)
            # searchsorted searches entire array; we need m < j for validity
            if m < j:
                candidate = suf[m] + 1
                k = suf_arg[m]
            else:
                candidate = 2
                k = -1
            val = max(2, candidate)
            dp[i, j] = val
            prev_idx[i, j] = k
            if val > best_len:
                best_len = val
                best_i = i
                best_j = j
    
    # Reconstruct path
    path = [int(primes[best_i])]
    cur_i = best_i
    cur_j = best_j
    while cur_j >= 0:
        path.append(int(primes[cur_j]))
        pj = int(prev_idx[cur_i, cur_j])
        cur_i = cur_j
        cur_j = pj
    
    path.reverse()
    return path

# EVOLVE-BLOCK-END
