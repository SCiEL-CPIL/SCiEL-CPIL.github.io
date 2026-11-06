"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def _precompute_small_part(limit):
    """Precompute the small-prime-part for all numbers 1..limit."""
    # small_part[x] = the part of x composed of primes strictly below 24
    # = x / (x with all factors of primes < 24 removed)
    arr = np.arange(1, limit + 1, dtype=np.int64)
    for p in PRIMES:
        # For each multiple of p, divide out all factors of p
        for m in range(p, limit + 1, p):
            while arr[m] % p == 0:
                arr[m] //= p
    # Now arr[x] = x with all small-prime factors removed
    # small_part[x] = original_x / arr[x]
    original = np.arange(1, limit + 1, dtype=np.int64)
    small_part = original // arr
    return small_part

def _score(n, small_part):
    """Number of distinct extracted factors for n+1..n+24."""
    factors = small_part[n+1:n+25]
    return len(set(factors.tolist()))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    limit = 1000024
    small_part = _precompute_small_part(limit)
    
    best_n = 1000
    best_score = _score(1000, small_part)
    
    if best_score == 1:
        return best_n
    
    # Scan all n from 1000 to 999976 (so n+24 <= 1000000)
    for n in range(1000, 999977):
        if time.time() >= deadline:
            break
        factors = small_part[n+1:n+25]
        sc = len(set(factors.tolist()))
        if sc < best_score:
            best_score = sc
            best_n = n
            if best_score == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
