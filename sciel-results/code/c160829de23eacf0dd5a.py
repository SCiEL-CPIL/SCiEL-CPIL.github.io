"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _build_prime_sieve(limit):
    """Sieve of Eratosthenes: return bytearray where sieve[p] is 1 if p is prime."""
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = 0
    return sieve

def construct():
    deadline = time.time() + 80.0
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for _, hi in gaps)
    sieve = _build_prime_sieve(max_val)

    # Extract primes for fast access
    primes = [p for p in range(2, max_val + 1) if sieve[p]]

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        g = hi - lo
        # Find smallest prime >= g (the threshold for lpf)
        threshold_prime = None
        for p in primes:
            if p >= g:
                threshold_prime = p
                break
        if threshold_prime is None:
            continue

        # Search from lo+1 to hi-1 for m where m has no prime factor < g
        # Equivalently: lpf[m] >= g, meaning m is not divisible by any prime < g
        small_primes = [p for p in primes if p < g]

        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any prime < g
            is_valid = True
            for p in small_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                result[i] = int(m)
                found = True
                break

    return result
# EVOLVE-BLOCK-END
