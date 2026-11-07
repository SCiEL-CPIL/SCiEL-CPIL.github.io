"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute LPF (least prime factor) for all integers in the relevant range
_min_val = min(lo for lo, hi in P['gaps'])
_max_val = max(hi for lo, hi in P['gaps'])
_range_size = _max_val - _min_val + 1

def _compute_lpf_table(lo, hi):
    """Compute least prime factor for each integer in [lo, hi].
    Returns a list where lpf[i] = least prime factor of (lo + i), or 0 if prime."""
    size = hi - lo + 1
    lpf = [0] * size  # 0 means prime (no prime factor found yet)
    
    # Sieve-based LPF computation
    # For each prime p, mark multiples
    for p in range(2, hi + 1):
        if p > hi:
            break
        # Check if p is prime by seeing if it was marked
        # Actually, let's use a proper sieve approach
        pass
    
    # Better: use a sieve to compute LPF
    # lpf[i] stores the least prime factor of (lo + i)
    # Initialize all to 0 (meaning prime)
    lpf = [0] * size
    
    # For each potential prime factor p starting from 2
    for p in range(2, hi + 1):
        # If p itself has a smaller factor, it's not prime, skip
        if p >= lo and lpf[p - lo] != 0:
            continue
        # p is prime; mark all multiples of p in [lo, hi]
        # First multiple of p >= lo
        first = ((lo + p - 1) // p) * p
        if first < lo:
            first = lo
        for m in range(first, hi + 1, p):
            idx = m - lo
            if lpf[idx] == 0:  # Only set if not already set (we want LEAST)
                lpf[idx] = p
    
    return lpf

_lpf_table = _compute_lpf_table(_min_val, _max_val)

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        # Check deadline
        if time.time() - _start_time > _DEADLINE:
            break
        
        gap = hi - lo
        
        # Find the integer in (lo, hi) with the highest LPF that is >= gap
        # A number m is valid if its LPF >= gap (or m is prime, LPF=0 treated as infinite)
        best_m = None
        best_lpf_val = -1
        
        for m in range(lo + 1, hi):
            idx = m - _min_val
            lp = _lpf_table[idx]
            # lp == 0 means m is prime, so its "least prime factor" is m itself
            # which is >= gap since m < hi and hi - lo = gap, so m > lo means m >= lo+1
            # Actually for a prime m, LPF is m itself. We need m >= gap.
            # Since m is in (lo, hi), m > lo. gap = hi - lo.
            # m >= gap is not always true. E.g., lo=10007, hi=10009, gap=2, m=10008.
            # For prime m, LPF = m. We need m >= gap.
            if lp == 0:
                effective_lpf = m
            else:
                effective_lpf = lp
            
            if effective_lpf >= gap:
                if effective_lpf > best_lpf_val:
                    best_lpf_val = effective_lpf
                    best_m = m
        
        if best_m is not None:
            result[i] = best_m
    
    # Ensure all entries are Python int or None
    return result

# EVOLVE-BLOCK-END
