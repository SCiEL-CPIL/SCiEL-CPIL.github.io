"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill

    def sieve(n):
        if n < 2:
            return []
        is_p = bytearray([1]) * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, n + 1) if is_p[i]]

    # Sieve up to max possible gap + 1 to cover all primes we might need
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes_small = sieve(max_gap + 1)

    # Precompute the set of forbidden primes for each distinct gap size
    gap_to_primes = {}
    for lo, hi in P['gaps']:
        g = hi - lo
        if g not in gap_to_primes:
            gap_to_primes[g] = tuple(p for p in primes_small if p < g)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue

        check_primes = gap_to_primes[gap]

        if len(check_primes) == 0:
            # No primes to check, any number works
            result[i] = int(lo + 1)
            continue

        # Segmented sieve: mark positions in [lo+1, hi-1] divisible by any p in check_primes
        size = gap - 1  # number of candidates: lo+1, lo+2, ..., hi-1
        if size <= 0:
            continue

        bad = bytearray(size)  # 0 = good, 1 = bad

        for p in check_primes:
            # Find first m in [lo+1, hi-1] with m % p == 0
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

        # Find the first good position (scanning from left)
        found = False
        for offset in range(size):
            if bad[offset] == 0:
                result[i] = int(lo + 1 + offset)
                found = True
                break

        if not found:
            # Direct verification fallback: check each candidate individually
            for m in range(lo + 1, hi):
                ok = True
                for p in check_primes:
                    if m % p == 0:
                        ok = False
                        break
                if ok:
                    result[i] = int(m)
                    found = True
                    break

    return result
# EVOLVE-BLOCK-END
