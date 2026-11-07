import math
from itertools import product

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _lcm_list(lst):
    from functools import reduce
    return reduce(_lcm, lst)

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _lcm_list(_DIVISORS)  # 30030
_MAX = 1000000

def _find_best_for_base(B, divisors):
    """For a base B (multiple of all divisors), find distinct multiples of each divisor
    near B minimizing range. Returns (values, range) or None."""
    n = len(divisors)
    # For each divisor, generate candidate multiples near B
    # We'll search offsets in increasing |offset|
    # Use a simple approach: try small ranges of multiples
    # For divisor d, multiples near B: B + t*d for t in some range
    # Since B is multiple of d, t=0 gives B.
    
    # We need to assign distinct values. Use backtracking with pruning.
    # To keep it fast, limit the search window.
    
    best = None
    # Try expanding window
    for half_width in range(0, 200):  # search in [B-half_width, B+half_width]
        lo = B - half_width
        hi = B + half_width
        if lo < 1 or hi > _MAX:
            if lo < 1:
                lo = 1
            if hi > _MAX:
                hi = _MAX
            if lo > hi:
                break
        
        # For each divisor, find all multiples of d in [lo, hi]
        candidates = []
        for d in divisors:
            # First multiple >= lo
            k_start = (lo + d - 1) // d
            k_end = hi // d
            if k_start > k_end:
                candidates.append([])
                continue
            muls = [k * d for k in range(k_start, k_end + 1)]
            candidates.append(muls)
        
        # Check if all have candidates
        if any(len(c) == 0 for c in candidates):
            continue
        
        # Backtracking to assign distinct values minimizing range
        # Since we want min range, and window is fixed, just check if a valid assignment exists
        # Then the range is at most hi - lo, but actual range might be smaller.
        # For efficiency, just try to find ANY valid assignment in this window.
        
        used = set()
        assignment = [None] * n
        
        def backtrack(i):
            if i == n:
                return True
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(i + 1):
                        return True
                    used.remove(val)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            vals = assignment[:]
            r = max(vals) - min(vals)
            if best is None or r < best[1]:
                best = (vals, r)
            break  # smallest half_width that works, with minimal range within it
    
    return best

def construct():
    divisors = _DIVISORS[:]
    best_result = None
    best_range = float('inf')
    
    # Search over bases B = k * LCM
    k_min = max(1, 1 // _LCM)
    k_max = _MAX // _LCM
    
    for k in range(k_min, k_max + 1):
        B = k * _LCM
        res = _find_best_for_base(B, divisors)
        if res is not None:
            vals, r = res
            if r < best_range:
                best_range = r
                best_result = vals[:]
    
    if best_result is None:
        # Fallback: just use the divisors themselves (valid but not optimal)
        return divisors[:]
    
    return [int(v) for v in best_result]
# EVOLVE-BLOCK-END
