"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

def _can_assign_fast(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor.
    Returns the assignment list or None."""
    n = len(divisors)
    # For each divisor i, compute candidate offsets p in [0, R-1] such that (L+p) % d_i == 0
    # p ≡ (-L) % d_i (mod d_i)
    valid_offsets = []
    for i in range(n):
        d = divisors[i]
        target = (-L) % d
        offsets = []
        if target < R:
            p = target
            while p < R:
                offsets.append(p)
                p += d
        if not offsets:
            return None
        valid_offsets.append(offsets)
    
    # MRV backtracking
    order = sorted(range(n), key=lambda i: len(valid_offsets[i]))
    assigned_offset = [-1] * n
    used_offsets = [False] * R
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for p in valid_offsets[i]:
            if not used_offsets[p]:
                used_offsets[p] = True
                assigned_offset[i] = p
                if backtrack(idx + 1):
                    return True
                used_offsets[p] = False
                assigned_offset[i] = -1
        return False
    
    if backtrack(0):
        return [L + assigned_offset[i] for i in range(n)]
    return None

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Search over R from small to large
    # For each R, iterate L over one LCM period
    # Key optimization: precompute for each R which L values are "promising"
    # by checking that each divisor has at least one multiple in the window.
    
    for R in range(1, 5000):
        if time.time() > deadline:
            break
        
        # For this R, we need to check L in [1, LCM]
        # But we can prune: for a given L, if any divisor has no multiple in [L, L+R-1], skip.
        # A multiple of d is in [L, L+R-1] iff there exists k such that L <= k*d <= L+R-1
        # i.e., k is in [ceil(L/d), floor((L+R-1)/d)]
        # This is non-empty iff ceil(L/d) <= floor((L+R-1)/d)
        # i.e., there exists a multiple of d in [L, L+R-1]
        
        # For efficiency, let's just iterate L and use the fast checker.
        # The fast checker already does early pruning.
        
        for L in range(1, _LCM + 1):
            if time.time() > deadline:
                break
            
            assigned = _can_assign_fast(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(x) for x in assigned]
    
    # Fallback: return the divisors themselves (may not be optimal but valid)
    # Actually, let's try a simple heuristic fallback
    # For each divisor, find a multiple near the median of the divisors
    result = []
    for d in divisors:
        # Find a multiple of d
        result.append(int(d))
    
    # Check if they are distinct
    if len(set(result)) == len(result):
        return result
    
    # If not distinct, use a different approach
    # Just return multiples with offsets
    base = 100
    result = []
    for i, d in enumerate(divisors):
        val = base + i * 10
        # Make it divisible by d
        val = ((val + d - 1) // d) * d
        result.append(int(val))
    
    return result

# EVOLVE-BLOCK-END
