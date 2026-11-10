"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Generate all primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(i) for i in np.where(sieve)[0]]

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Generate primes up to 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # DP: dp[i] = length of longest valid sequence ending at primes[i]
    # prev[i] = index of previous element in optimal sequence ending at i
    dp = np.zeros(n, dtype=np.int32)
    prev = np.full(n, -1, dtype=np.int32)
    
    for i in range(n):
        dp[i] = 1
        pi = primes[i]
        for j in range(i):
            pj = primes[j]
            gap = pi - pj
            if j == 0:
                # Sequence of length 2 is always valid
                if 2 > dp[i]:
                    dp[i] = 2
                    prev[i] = j
            else:
                prev_gap = pj - primes[prev[j]]
                if gap >= prev_gap:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j
        
        if time.time() > deadline:
            break
    
    # Find the index with maximum dp value
    best_end = int(np.argmax(dp))
    
    # Reconstruct the sequence
    result = []
    idx = best_end
    while idx != -1:
        result.append(primes[idx])
        idx = prev[idx]
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END
