"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'n': 256}

def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _largest_prime_power(p, n):
    """Return largest power of p that is <= n."""
    result = p
    while result * p <= n:
        result *= p
    return result

def construct():
    n = P['n']
    primes = _sieve(n)
    
    # For each prime, pick the largest power <= n
    selected = []
    for p in primes:
        selected.append(_largest_prime_power(p, n))
    
    # Include 1 (it's coprime with everything)
    result = [1] + selected
    
    # Ensure all are Python ints and distinct
    result = [int(x) for x in result]
    # They should be distinct since each has a unique prime factor
    return result
# EVOLVE-BLOCK-END
