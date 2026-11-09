"""Rewritten program with optimized local trial division for witness search."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    """
    Finds integer witnesses m in (lo, hi) such that least_prime_factor(m) >= gap.
    Optimized by checking divisibility by small primes only.
    """
    deadline = time.time() + 55.0
    
    # 1. Determine the maximum gap to identify relevant small primes
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # 2. Generate all primes strictly less than max_gap
    # These are the only primes that could cause lpf(m) < gap
    small_primes = []
    for i in range(2, max_gap):
        is_prime = True
        for p in small_primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            small_primes.append(i)
            
    result = [None] * len(P['gaps'])
    
    # 3. Search for witnesses in each gap
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Filter primes that are strictly less than the current gap
        # If m is divisible by any of these, lpf(m) <= p < gap, so m is invalid.
        # If m is not divisible by any of these, lpf(m) >= gap (either m is prime or its smallest factor is >= gap).
        relevant_primes = [p for p in small_primes if p < gap]
        
        witness = None
        # Iterate through integers strictly between lo and hi
        for m in range(lo + 1, hi):
            is_valid = True
            for p in relevant_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                witness = m
                break
                
        if witness is not None:
            result[i] = int(witness)
            
    return result
# EVOLVE-BLOCK-END
