"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm(lst):
    r = 1
    for x in lst:
        r = _lcm(r, x)
    return r

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _compute_lcm(_DIVS)  # 30030
_N = len(_DIVS)
_MAX_VAL = 1000000

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill

    # Precompute for all s in [1, LCM] (s=0 gives non-positive multiples for some divisors)
    # For starting position s >= 1:
    #   first multiple of d in [s, s+W] is d * ceil(s/d)
    #   offset = d * ceil(s/d) - s
    #   valid iff offset <= W
    #
    # We'll use s in [1, LCM] to ensure all multiples are positive.
    
    LCM_START = 1
    LCM_END = _LCM + 1  # exclusive
    
    max_offset = [0] * _LCM
    multiples = [[0] * _N for _ in range(_LCM)]
    
    for idx, s in enumerate(range(LCM_START, LCM_END)):
        mo = 0
        for i, d in enumerate(_DIVS):
            # d * ceil(s / d)
            q = (s + d - 1) // d
            m = d * q
            off = m - s
            multiples[idx][i] = m
            if off > mo:
                mo = off
        max_offset[idx] = mo

    # Sort indices by max_offset ascending
    sorted_idx = sorted(range(_LCM), key=lambda i: max_offset[i])
    
    best_result = None
    best_W = None

    for W in range(1, 200):
        if time.time() > deadline:
            break

        found = False
        for idx in sorted_idx:
            if max_offset[idx] > W:
                break  # sorted, so no more candidates

            vals = multiples[idx]
            
            # Check distinctness
            if len(set(vals)) != _N:
                continue

            # Check all <= MAX_VAL
            if max(vals) > _MAX_VAL:
                continue

            # All positive by construction (s >= 1, d >= 1)
            
            # Valid solution found
            found = True
            best_W = W
            best_result = vals
            break

        if found:
            break

    if best_result is not None:
        return [int(x) for x in best_result]

    # Fallback: return the divisors themselves
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
