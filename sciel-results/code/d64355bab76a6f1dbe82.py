"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

DEADLINE = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _sieve_spf(max_n):
    """Build smallest prime factor array up to max_n using vectorized sieve."""
    spf = np.arange(max_n + 1, dtype=np.int64)
    if max_n >= 0:
        spf[0] = 0
    if max_n >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            # Vectorized: mark multiples of i starting from i*i
            start = i * i
            step = i
            end = max_n + 1
            indices = np.arange(start, end, step)
            # Only update where spf is still the number itself (i.e., not yet marked by smaller prime)
            mask = spf[indices] == indices
            spf[indices[mask]] = i
    return spf

def _precompute_powerful_parts(max_n, spf):
    """Precompute powerful part for all numbers 1..max_n using SPF-based factorization."""
    pp = np.ones(max_n + 1, dtype=np.int64)
    for n in range(2, max_n + 1):
        if _elapsed() > DEADLINE:
            break
        temp = n
        ppart = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                ppart *= p ** e
        pp[n] = ppart
    return pp

def _precompute_powerful_parts_fast(max_n):
    """Vectorized sieve: for each prime power p^k (k>=2), multiply pp[multiples] by p^k."""
    pp = np.ones(max_n + 1, dtype=np.int64)
    # Sieve to find primes
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    for p in primes:
        if _elapsed() > DEADLINE:
            break
        # Generate prime powers p^k for k >= 2
        power = p * p
        k = 2
        while power <= max_n and k <= 20:
            # Multiply pp at all multiples of power by power
            indices = np.arange(power, max_n + 1, power)
            pp[indices] *= power
            k += 1
            power *= p
    return pp

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    max_n = 1000000
    min_n = 10000
    
    # Use fast vectorized sieve to compute powerful parts
    pp = _precompute_powerful_parts_fast(max_n + 1)
    if _elapsed() > DEADLINE:
        return 10000
    
    # Direct scan of all consecutive pairs
    scores = pp[min_n:max_n+1] * pp[min_n+1:max_n+2]
    idx = int(np.argmax(scores))
    best_score = int(scores[idx])
    best_n = min_n + idx
    
    if _elapsed() > DEADLINE:
        return best_n
    
    # Local refinement around the best candidate
    # Check neighbors of best_n for potential improvements
    for delta in range(-50, 51):
        if _elapsed() > DEADLINE:
            break
        n_check = best_n + delta
        if n_check < min_n or n_check > max_n:
            continue
        s = int(pp[n_check] * pp[n_check + 1])
        if s > best_score:
            best_score = s
            best_n = n_check
    
    return best_n

# EVOLVE-BLOCK-END
