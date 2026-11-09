"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    DEADLINE = 85

    # Determine the full range we need to cover
    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])

    # Precompute least prime factor (LPF) for every number in [all_lo, all_hi]
    # using a sieve approach: for each prime p, mark multiples with p if not yet marked
    size = all_hi - all_lo + 1
    lpf = [0] * size  # 0 means "not yet assigned" (will be prime or unmarked)

    # Sieve: for each prime p, assign lpf to its multiples
    # We need primes up to sqrt(all_hi) for factoring, but for LPF we go up to all_hi
    # Actually, for LPF we only need to check primes up to sqrt(n) for each n,
    # but a more efficient approach: iterate primes and mark their multiples
    max_p = isqrt(all_hi) + 1
    small_primes = _primes_up_to(max_p)

    # For each prime p, mark all multiples of p in [all_lo, all_hi] with lpf = p (if not already set)
    for p in small_primes:
        # First multiple of p >= all_lo
        first = ((all_lo + p - 1) // p) * p
        for m in range(first, all_hi + 1, p):
            idx = m - all_lo
            if lpf[idx] == 0:
                lpf[idx] = p

    # For numbers where lpf is still 0, they are prime (or 1, but our range is > 2)
    # Their LPF is themselves
    for i in range(size):
        if lpf[i] == 0:
            lpf[i] = all_lo + i  # the number itself is its least prime factor

    # Now for each gap, find a witness
    n = len(P['gaps'])
    result = [None] * n

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - start_time > DEADLINE:
            break
        gap = hi - lo
        # Candidates are integers strictly between lo and hi: lo+1, lo+2, ..., hi-1
        for m in range(lo + 1, hi):
            idx = m - all_lo
            if lpf[idx] >= gap:
                result[i] = int(m)
                break
        # If no witness found, result[i] stays None

    # Ensure all are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
