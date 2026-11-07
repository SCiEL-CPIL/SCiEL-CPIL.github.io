# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Determine the full range we need to cover
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])

    # Generate small primes up to max_gap for LPF computation
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _gen_primes(max_gap)

    # Compute LPF for all integers in [min_val, max_val]
    range_size = max_val - min_val + 1
    lpf = [0] * range_size

    # Initialize: LPF[n] = n itself (assume prime until proven otherwise)
    for i in range(range_size):
        lpf[i] = min_val + i

    # Sieve: for each small prime in ascending order, mark its multiples
    for p in small_primes:
        first_mult = ((min_val + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - min_val
            if lpf[idx] > p:
                lpf[idx] = p

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        g = hi - lo

        # Scan the interior of the gap for a witness with LPF >= gap
        for m in range(lo + 1, hi):
            idx = m - min_val
            if lpf[idx] >= g:
                result[i] = m
                break

    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))
    return final_result
# EVOLVE-BLOCK-END
