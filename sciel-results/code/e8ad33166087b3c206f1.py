"""Rewritten program: precompute LPF via linear sieve, then scan each gap for witnesses."""

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _build_lpf(limit):
    """Linear sieve to compute least prime factor for all integers up to limit."""
    lpf = list(range(limit + 1))
    if limit >= 0:
        lpf[0] = 0
    if limit >= 1:
        lpf[1] = 1
    primes = []
    for i in range(2, limit + 1):
        if lpf[i] == i:
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > limit:
                break
            lpf[i * p] = p
    return lpf


def construct():
    max_prime = max(hi for _, hi in P['gaps'])
    lpf = _build_lpf(max_prime)

    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)
                break
    return result
