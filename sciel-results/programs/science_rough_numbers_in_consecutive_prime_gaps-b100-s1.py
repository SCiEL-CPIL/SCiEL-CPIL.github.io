# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _least_prime_factor(m, small_primes):
    """Return least prime factor of m, or m if prime."""
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            return p
    # Check if m itself is prime by testing up to sqrt(m)
    if m <= 1:
        return m
    for d in range(small_primes[-1] + 1, isqrt(m) + 1):
        if m % d == 0:
            return d
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Precompute small primes up to 200 (more than enough for our needs)
    limit = 200
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Search for a witness m in (lo, hi) with LPF(m) >= gap
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Find least prime factor
            lpf = m  # default: m is prime
            for p in small_primes:
                if p * p > m:
                    break
                if m % p == 0:
                    lpf = p
                    break
            if lpf >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
