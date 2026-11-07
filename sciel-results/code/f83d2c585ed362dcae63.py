"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def sieve(n):
        if n < 2:
            return []
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = 0
        is_p[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, n + 1) if is_p[i]]

    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes_small = sieve(max_gap + 1)

    gap_to_primes = {}
    for lo, hi in P['gaps']:
        g = hi - lo
        if g not in gap_to_primes:
            gap_to_primes[g] = [p for p in primes_small if p < g]

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue

        check_primes = gap_to_primes[gap]
        size = gap - 1
        bad = bytearray(size)

        for p in check_primes:
            rem = (lo + 1) % p
            if rem == 0:
                first = lo + 1
            else:
                first = lo + 1 + (p - rem)
            if first >= hi:
                continue
            start_offset = first - (lo + 1)
            for k in range(start_offset, size, p):
                bad[k] = 1

        for offset in range(size):
            if bad[offset] == 0:
                result[i] = int(lo + 1 + offset)
                break

    return [int(v) if v is not None else None for v in result]
# EVOLVE-BLOCK-END
