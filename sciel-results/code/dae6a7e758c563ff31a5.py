import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    min_lo = min(lo for lo, hi in gaps)
    max_hi = max(hi for lo, hi in gaps)
    offset = min_lo
    span = max_hi - min_lo + 1

    # Segmented LPF computation for [min_lo, max_hi]
    # lpf[0] corresponds to min_lo, lpf[span-1] corresponds to max_hi
    lpf = list(range(span))
    for i in range(span):
        val = min_lo + i
        lpf[i] = val

    # Sieve: for each prime p up to isqrt(max_hi), mark lpf for composites
    for p in range(2, isqrt(max_hi) + 1):
        # Find first multiple of p in [min_lo, max_hi]
        first = ((min_lo + p - 1) // p) * p
        if first < p * p:
            first = p * p
        for multiple in range(first, max_hi + 1, p):
            idx = multiple - min_lo
            if lpf[idx] == min_lo + idx:  # not yet marked
                lpf[idx] = p

    # For each gap, find witness m in (lo, hi) with lpf[m] >= gap
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        for m in range(lo + 1, hi):
            idx = m - min_lo
            if lpf[idx] >= gap:
                result[i] = int(m)
                break

    return result
# EVOLVE-BLOCK-END
