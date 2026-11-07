"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin below the 90s wall-clock limit

    # Range we need to cover
    lo_bound = min(lo for lo, hi in P['gaps'])
    hi_bound = max(hi for lo, hi in P['gaps'])
    n = hi_bound

    # Least-prime-factor sieve for every integer up to n.
    # lpf[x] = smallest prime dividing x (== x if x is prime).
    lpf = list(range(n + 1))
    if n >= 0:
        lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    i = 2
    while i * i <= n:
        if lpf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if lpf[j] == j:
                    lpf[j] = i
        i += 1

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break

        gap = hi - lo
        witness = None
        # Scan the open interval (lo, hi) for an integer whose LPF >= gap.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                witness = m
                break
        result[i] = witness

    # Types are checked exactly by the verifier: ensure plain Python ints / None.
    return [int(w) if w is not None else None for w in result]

# EVOLVE-BLOCK-END
