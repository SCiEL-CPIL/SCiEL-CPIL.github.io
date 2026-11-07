"""Rewritten program for the suite task; stronger algorithm for finding witnesses."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(limit):
    """Return bytearray where is_prime[i] is 1 iff i is prime, for 0<=i<=limit."""
    if limit < 2:
        return bytearray(limit + 1)
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = bytearray(len(range(i*i, limit+1, i)))
    return is_prime

def _small_primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sp = _sieve(n)
    return [i for i in range(2, n + 1) if sp[i]]

def _build_residue_sieve(g, primes):
    """
    For a given gap size g and list of primes p < g, build a periodic boolean
    array of length L = product of all such primes (or a manageable period)
    where sieve[r] == 1 iff r is NOT divisible by any prime in primes.
    We use the Chinese Remainder Theorem-friendly approach: just mark multiples.
    Since g <= 28, the primes involved are small and the period is manageable.
    """
    if not primes:
        return None, 1
    # The period is the product of all distinct primes < g
    # For g up to 28, primes are 2,3,5,7,11,13,17,19,23
    # Product can be large, but we only need a window of size g-1
    # Actually, we can just build a sieve over a range of size lcm or product
    # But simpler: build a boolean array for residues 0..(period-1)
    period = 1
    for p in primes:
        period *= p
    # For g=28, primes < 28 are 2,3,5,7,11,13,17,19,23
    # Product = 2*3*5*7*11*13*17*19*23 = 223092870 — too large!
    # Instead, use a different approach: for each candidate m, check m % p for p in primes
    # But we can precompute for a smaller period. Since we only need to check
    # m mod p for each p, and the condition is periodic with period = lcm(primes) = product
    # (since they're distinct primes), we can't avoid the large period.
    # 
    # Alternative: precompute a "forbidden" bitmask for each residue class mod each prime,
    # then for a candidate m, check all at once. But that's what we're already doing.
    #
    # Better idea: For each gap size g, precompute the set of valid residues mod each prime,
    # then for a candidate, just check membership. But that's equivalent to the current approach.
    #
    # Let's try a different tactic: precompute a numpy array for a range of size say 10000
    # that marks which numbers are "clean" (not divisible by any prime < g), then for each
    # interval (lo, hi), just look up the relevant slice.
    return None, 1

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    result = [None] * len(P['gaps'])
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _small_primes_up_to(max_gap)
    
    # Precompute primes below each possible gap size
    primes_below = {g: [p for p in small_primes if p < g] for g in range(2, max_gap + 1)}
    
    # Precompute a "clean" sieve for a large range using numpy for speed
    # We'll sieve up to a value that covers all our intervals
    max_val = max(hi for lo, hi in P['gaps'])
    min_val = min(lo for lo, hi in P['gaps'])
    range_size = max_val - min_val + 1
    
    # For each distinct gap size, create a boolean array indicating which offsets
    # from min_val are "clean" for that gap size
    # But actually, cleanliness depends on the absolute value m, not just offset.
    # m is clean for gap g iff for all primes p < g, m % p != 0.
    # This is periodic with period = product of primes < g.
    # 
    # Since the gaps are small (<=28), the relevant primes are small.
    # Let's precompute for each distinct gap size g, a numpy boolean array
    # of length = period_g, where arr[r] = 1 iff r is not divisible by any prime < g.
    
    distinct_gaps = set(hi - lo for lo, hi in P['gaps'])
    gap_sieves = {}
    for g in distinct_gaps:
        p_list = primes_below.get(g, [])
        if not p_list:
            gap_sieves[g] = (None, 1)  # Any value works
            continue
        # Build a sieve for one period
        period = 1
        for p in p_list:
            period *= p
        # For g <= 28, period can be up to ~223M which is too large.
        # Instead, just use the direct check but vectorize with numpy for the interval.
        gap_sieves[g] = (p_list, period)
    
    for i, (lo, hi) in enumerate(P['gaps']):
        g = hi - lo
        p_list = primes_below.get(g, [])
        
        if not p_list:
            result[i] = int(lo + 1)
            continue
        
        # Use numpy to vectorize the check over the interval
        import numpy as np
        candidates = np.arange(lo + 1, hi)
        if len(candidates) == 0:
            continue
        
        # Check each prime: mark candidates divisible by p
        valid = np.ones(len(candidates), dtype=bool)
        for p in p_list:
            valid &= (candidates % p != 0)
        
        if np.any(valid):
            first_valid_idx = np.argmax(valid)
            result[i] = int(candidates[first_valid_idx])
        
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
