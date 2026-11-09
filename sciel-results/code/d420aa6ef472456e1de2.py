"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _least_prime_factor(m, primes):
    """Return the least prime factor of m using precomputed primes list."""
    if m < 2:
        return m
    for p in primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m


def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s (evaluator kills at 90s, but contract says 60s)
    
    # Determine the maximum gap size to know how many primes we need
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need primes up to sqrt(max_candidate) where max_candidate is the largest number in any gap
    max_candidate = max(hi for lo, hi in P['gaps'])
    # For least prime factor check, we need primes up to sqrt(max_candidate)
    # But actually, if the least prime factor is >= gap, and gap can be up to ~40,
    # we need to check divisibility by all primes up to sqrt(m)
    # sqrt(10289) ≈ 101.4, so we need primes up to ~102
    # But to be safe and fast, let's sieve up to a reasonable bound
    # Actually, for the check "least prime factor >= gap", we need to verify that
    # no prime < gap divides m. If m has no prime factor < gap, then its least prime
    # factor is either >= gap or m itself is prime (in which case LPF = m >= gap since gap >= 2).
    # So we only need primes less than the gap size for each gap.
    
    # Let's precompute primes up to a safe upper bound
    # The largest gap is 10289-10273 = 16, wait let me check...
    # 10009-10007=2, 10037-10009=28, 10039-10037=2, 10061-10039=22, ...
    # Let me just compute max_gap properly
    max_gap_val = max(hi - lo for lo, hi in P['gaps'])
    
    # We need primes up to max_gap_val (exclusive) for the divisibility check
    # If a number m in (lo, hi) has no prime factor < (hi-lo), then LPF(m) >= hi-lo
    # So we need primes < max_gap_val
    primes_needed = _sieve_primes_up_to(max_gap_val - 1) if max_gap_val > 2 else []
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find primes less than gap for this specific gap
        # We can filter our precomputed list
        small_primes = [p for p in primes_needed if p < gap]
        
        # Search for a witness m in (lo, hi) such that LPF(m) >= gap
        # This means m is not divisible by any prime < gap
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Check if m is divisible by any prime < gap
            is_valid = True
            for p in small_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                result[i] = m
                break
    
    # Ensure all entries are Python int or None
    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))
    
    return final_result
# EVOLVE-BLOCK-END
