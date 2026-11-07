"""Combined: global LPF sieve + per-gap selection of highest-LPF witness."""
import time
_start_time = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(n):
    """Return list lp where lp[m] is the least prime factor of m, or m itself if prime."""
    if n < 2:
        return list(range(n + 1))
    lp = list(range(n + 1))
    lp[0] = 0
    lp[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if lp[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lp[j] == j:
                    lp[j] = i
    return lp

# Precompute LPF for all integers in the range of interest
_max_val = max(hi for _, hi in P['gaps'])
_lpf = _sieve_lpf(_max_val)

def _find_witness(lo, hi):
    """Find an integer m in (lo, hi) whose least prime factor is >= gap = hi - lo.
    
    Uses the precomputed LPF array for O(1) lookup per candidate.
    Selects the candidate with the highest LPF for robustness.
    """
    gap = hi - lo
    if gap <= 1:
        return None
    
    best_m = None
    best_lpf_val = 0
    
    for m in range(lo + 1, hi):
        lpf_val = _lpf[m]
        if lpf_val >= gap:
            if lpf_val > best_lpf_val:
                best_lpf_val = lpf_val
                best_m = m
    
    return best_m

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
        w = _find_witness(lo, hi)
        if w is not None:
            result[i] = int(w)
    
    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
