import numpy as np
import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = start_time + 85.0

    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # Pre-allocate 2D arrays for better performance
    # DP[i, g] = max length of sequence ending at primes[i] with last gap <= g
    DP = np.zeros((n, LIMIT + 1), dtype=np.int32)
    # ARG[i, g] = index h in [0..g] such that exact_dp[h] is max for prime i
    ARG = np.zeros((n, LIMIT + 1), dtype=np.int32)
    # PAR[i, g] = parent prime index for the transition that set exact_dp[g] at i
    PAR = np.full((n, LIMIT + 1), -1, dtype=np.int32)
    # PGAP[i, g] = prev gap index for the transition that set exact_dp[g] at i
    PGAP = np.zeros((n, LIMIT + 1), dtype=np.int32)
    
    best_len_global = 1
    best_end_i = 0
    best_end_gap = 0
    
    for i in range(n):
        # exact_dp is a view or temporary array?
        # We need to compute exact_dp for i, then prefix max.
        # exact_dp[g] is the max length ending at i with EXACT last gap g.
        # We can store this in a temporary array, then compute prefix max into DP[i].
        exact_dp = np.ones(LIMIT + 1, dtype=np.int32)
        
        if time.time() > DEADLINE:
            break

        for j in range(i):
            gap = primes[i] - primes[j]
            # Query best length ending at j with last gap <= gap
            len_prev = DP[j, gap]
            len_curr = 1 + len_prev
            
            if len_curr > exact_dp[gap]:
                exact_dp[gap] = len_curr
                PAR[i, gap] = j
                PGAP[i, gap] = ARG[j, gap]
        
        # Build prefix max for current i
        DP[i, :] = np.maximum.accumulate(exact_dp)
        
        # Compute ARG[i, g]
        # ARG[i, g] is the smallest h in [0..g] such that exact_dp[h] == DP[i, g]
        # We can compute this by iterating g
        best_g = 0
        best_v = exact_dp[0]
        ARG[i, 0] = 0
        for g in range(1, LIMIT + 1):
            if exact_dp[g] > best_v:
                best_v = exact_dp[g]
                best_g = g
            ARG[i, g] = best_g
            
        # Update global best
        max_val = DP[i, LIMIT]
        if max_val > best_len_global:
            best_len_global = max_val
            best_end_i = i
            best_end_gap = ARG[i, LIMIT]
        
        if time.time() > DEADLINE:
            break

    # Reconstruct sequence
    if best_len_global == 1:
        return [primes[0]]
        
    seq = []
    i = best_end_i
    g = best_end_gap
    
    while i >= 0:
        seq.append(primes[i])
        if i == 0:
            break
        j = PAR[i, g]
        if j < 0:
            # Fallback if reconstruction fails
            break
        g_prev = PGAP[i, g]
        i = j
        g = g_prev
        
    seq.reverse()
    return seq
