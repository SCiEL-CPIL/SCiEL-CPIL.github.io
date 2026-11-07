"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
from math import isqrt

def _sigma(n, pf):
    """Compute sigma(n) given prime factorization pf as list of (p, e)."""
    result = 1
    for p, e in pf:
        result *= (p**(e+1) - 1) // (p - 1)
    return result

def _gen_smooth(primes, max_exp, limit):
    """Generate all numbers <= limit that are products of primes in 'primes' with exponents <= max_exp."""
    nums = [1]
    for p in primes:
        new_nums = []
        for n in nums:
            val = n
            for e in range(1, max_exp + 1):
                val *= p
                if val > limit:
                    break
                new_nums.append(val)
        nums.extend(new_nums)
    return nums

def _factorize(n, primes):
    """Factorize n using the given primes list. Returns (pf, ok) where pf is list of (p,e)."""
    pf = []
    for p in primes:
        if p * p > n:
            break
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e > 0:
            pf.append((p, e))
    if n > 1:
        # n must be prime or 1; if it's not in our prime list, we can't fully factor
        # but for our generated numbers it should be fine
        pf.append((n, 1))
    return pf, True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Use primes up to ~200 for good coverage
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    # Generate smooth numbers with limited exponents
    # M <= 10^18, k <= sqrt(M) <= 10^9
    # k * sigma(k) = M, so k <= 10^9
    max_k = 10**9
    
    # Generate candidate k values that are smooth
    # Use max_exp=6 for small primes, less for larger
    smooth_k = _gen_smooth(primes[:20], 6, max_k)
    smooth_k = sorted(set(smooth_k))
    
    # Map k -> k*sigma(k)
    m_to_ks = defaultdict(list)
    
    for k in smooth_k:
        if time.time() > deadline:
            break
        pf, _ = _factorize(k, primes)
        s = _sigma(k, pf)
        m = k * s
        if m <= 10**18:
            m_to_ks[m].append(k)
    
    # Find M with most k solutions
    best_m = None
    best_ks = []
    for m, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = m
            best_ks = ks
        if time.time() > deadline:
            break
    
    if best_m is None or len(best_ks) < 3:
        # Fallback to known good example
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify and truncate to at most 1000
    best_ks = best_ks[:1000]
    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
# EVOLVE-BLOCK-END
