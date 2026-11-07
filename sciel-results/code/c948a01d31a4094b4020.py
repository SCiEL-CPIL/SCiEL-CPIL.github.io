"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _least_prime_factor(n):
    """Return the least prime factor of n, or n if n is prime."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    # Check divisors of form 6k ± 1
    d = 5
    while d * d <= n:
        if n % d == 0:
            return d
        if n % (d + 2) == 0:
            return d + 2
        d += 6
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to a reasonable bound for quick factorization
    # The primes we need to check are small (up to ~max gap which is ~30)
    # Actually, we need to check if the least prime factor >= gap.
    # For a number m in (lo, hi), its least prime factor must be >= gap.
    # This means m is not divisible by any prime < gap.
    
    # Strategy: For each gap, iterate through candidates m in (lo, hi).
    # For each m, check if it has any prime factor < gap.
    # If not, it's a valid witness.
    
    # Precompute small primes up to max possible gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = []
    n = 2
    while n < max_gap:
        is_prime = True
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            small_primes.append(n)
        n += 1
    
    # Sort gaps by size (larger gaps are easier to find witnesses for)
    # Actually, let's just process all gaps. We want to maximize count.
    # Let's try to find a witness for each gap.
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap
        # This means m is not divisible by any prime p < gap
        
        # Generate primes less than gap
        primes_below_gap = [p for p in small_primes if p < gap]
        
        # Try candidates in the gap
        # Heuristic: try from the middle out, or just from lo+1
        # Let's try a few candidates. Since gaps are small, this should be fast.
        
        found = False
        # Try all candidates in the gap
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            valid = True
            for p in primes_below_gap:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        if not found:
            # If no witness found, leave as None
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
