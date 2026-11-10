import time
import math
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 60.0

    N = 1000000
    NMAX = N + 1

    # --- Sieve primes up to NMAX ---
    # Using numpy for speed
    is_prime = np.ones(NMAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    limit = math.isqrt(NMAX)
    for i in range(2, limit + 1):
        if is_prime[i]:
            is_prime[i*i : NMAX+1 : i] = False

    # --- Largest prime factor for every integer up to NMAX ---
    # Vectorized update
    lpf = np.zeros(NMAX + 1, dtype=np.int32)
    
    # Get all primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Update lpf for multiples of p
        # This is a vectorized slice assignment
        lpf[p : NMAX+1 : p] = p

    # --- Scan the window ---
    # We need max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # lpf[n] is at index n.
    # We want indices 10000 to 1000000 for n.
    # So we look at lpf[10000:1000001] and lpf[10001:1000002]
    
    left = lpf[10000 : N+1]
    right = lpf[10001 : N+2]
    
    pair_max = np.maximum(left, right)
    
    min_val = np.min(pair_max)
    min_idx = np.argmin(pair_max)
    
    best_n = 10000 + min_idx
    
    # Check deadline just in case, though this should be fast
    if time.time() > DEADLINE:
         # Fallback to a known good value or the first one
         return 10000

    return int(best_n)
