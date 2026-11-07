"""Rewritten program: segmented-sieve LPF over the full range, then scan each gap."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _primes_upto(n):
    """Return list of primes <= n (Sieve of Eratosthenes)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            start = i * i
            step = i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def _lpf_direct(n, small_primes):
    """Find least prime factor of n. Returns n if prime."""
    for p in small_primes:
        if p * p > n:
            return n
        if n % p == 0:
            return p
    return n


def construct():
    import time
    deadline = time.time() + 80.0  # safety margin vs 90s wall-clock

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # Determine the full range we need to cover
    lo_min = min(lo for lo, hi in gaps)
    hi_max = max(hi for lo, hi in gaps)

    # We need primes up to sqrt(hi_max) for LPF computation
    import math
    limit = int(math.isqrt(hi_max)) + 1
    small_primes = _primes_upto(limit)

    # For each gap, scan candidates and compute LPF directly
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        found = None
        for m in range(lo + 1, hi):
            lpf = _lpf_direct(m, small_primes)
            if lpf >= gap:
                found = m
                break
        if found is not None:
            result[i] = int(found)

        if time.time() > deadline:
            break

    # Ensure all entries are Python int or None
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
