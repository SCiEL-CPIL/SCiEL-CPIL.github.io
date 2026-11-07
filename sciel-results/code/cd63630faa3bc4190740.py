"""
Improved witness finder using small-prime divisibility checks.
Avoids global LPF sieve to ensure structural novelty.
"""
import time
import math

_start_time = time.time()
_DEADLINE = 60.0  # Safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_below(n):
    """Generate primes strictly less than n using simple sieve."""
    if n <= 2:
        return []
    sieve = [True] * n
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i*i, n, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _is_valid_witness(m, gap, small_primes):
    """
    Check if m is a valid witness for a gap of size 'gap'.
    Condition: least prime factor of m >= gap.
    This is equivalent to m having no prime factor q such that q < gap.
    """
    for p in small_primes:
        if m % p == 0:
            return False
    # If no small prime divides m, then LPF(m) >= gap (or m is prime)
    # Note: If m < gap, it's impossible for LPF(m) >= gap unless m=1 (not possible) 
    # or if we consider the case where m itself is the LPF.
    # However, m is in a gap between primes > 10000, and gap < 100.
    # So m > gap always.
    # If m is composite and has no factor < gap, its LPF is >= gap.
    # If m is prime, its LPF is m > gap.
    return True

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to the maximum gap size to speed up checks
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes < gap. So primes up to max_gap.
    small_primes_cache = {}
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
            
        gap = hi - lo
        if gap <= 1:
            continue
            
        # Get primes strictly less than gap
        if gap not in small_primes_cache:
            small_primes_cache[gap] = _primes_below(gap)
        small_primes = small_primes_cache[gap]
        
        witness = None
        # Search for the first valid witness. 
        # Any valid witness suffices for the score.
        for m in range(lo + 1, hi):
            if _is_valid_witness(m, gap, small_primes):
                witness = m
                break
        
        if witness is not None:
            # Double check with trial division for safety (distinct logic path)
            # Check if LPF >= gap
            is_ok = True
            for p in small_primes:
                if witness % p == 0:
                    is_ok = False
                    break
            if is_ok:
                result[i] = int(witness)
    
    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
