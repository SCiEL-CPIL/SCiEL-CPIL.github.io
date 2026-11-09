# EVOLVE-BLOCK-START
import math
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _lpf_sieve(n):
    lpf = [0] * (n + 1)
    for i in range(2, n + 1):
        if lpf[i] == 0:
            lpf[i] = i
            if i * i <= n:
                for j in range(i * i, n + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i
    return lpf

def construct():
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    t0 = time.time()
    deadline = t0 + 85.0

    max_hi = max(hi for _, hi in gaps)
    lpf = _lpf_sieve(max_hi)

    for idx, (lo, hi) in enumerate(gaps):
        if time.time() > deadline - 2.0:
            break
        gap_size = hi - lo
        
        # Try to find a witness
        found = False
        for m in range(lo + 1, hi):
            # Check LPF condition
            f = lpf[m]
            if f == 0:
                f = m
            if f >= gap_size:
                result[idx] = int(m)
                found = True
                break
        
    # Enforce exact types
    out = []
    for v in result:
        if v is None:
            out.append(None)
        else:
            out.append(int(v))
    return out

# EVOLVE-BLOCK-END
