"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    deadline = time.time() + 80.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for _, hi in gaps)

    # Linear sieve for least prime factor (SPF) up to max_val
    spf = list(range(max_val + 1))
    spf[0] = 0
    spf[1] = 1
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > max_val:
                break
            spf[i * p] = p

    # For each gap, find a valid witness m in (lo, hi) with spf[m] >= gap_size
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        if gap_size <= 1:
            continue
        # Check all interior integers; pick first valid one
        for m in range(lo + 1, hi):
            if spf[m] >= gap_size:
                result[i] = int(m)
                break

    return result

# EVOLVE-BLOCK-END
