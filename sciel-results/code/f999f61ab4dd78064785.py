"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

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

# Identify "rigid" divisors (d > W_max_expected) vs "flexible"
# In a tight window (W ~ 10-20), divisors > W have at most 1 multiple.
# Divisors <= W might have 2+.
# Let's classify based on a threshold, say W=20.
# Divisors > 20: 21, 22, 26, 33, 35, 39. These are rigid.
# Divisors <= 20: 6, 10, 14, 15. These are flexible.

# We will iterate over the rigid divisors' residues mod LCM?
# No, that's 30030^6.
# Instead, iterate over the window start L mod LCM.
# For a fixed L and W, the rigid divisors have a fixed candidate (if any).
# The flexible divisors have a small set of candidates.
# We can check if the rigid candidates are distinct and within [L, L+W].
# Then we need to assign the flexible divisors to the remaining slots
# such that all are distinct and within [L, L+W].

# Optimization:
# 1. Precompute for each divisor d, the list of residues r in [0, LCM) such that r % d == 0.
#    Actually, we just need to find the multiple in [L, L+W].
#    For a fixed L, the candidate for d is:
#    m = L + ((d - (L % d)) % d)
#    If m > L + W, then no candidate.
#    If d > W, there is at most 1 candidate.
#    If d <= W, there might be 2 (e.g., d=6, W=10, L=0 -> 0, 6).

# Strategy:
# Iterate W from 1 upwards.
# For each W, iterate L from 0 to LCM-1.
# For each L:
#   1. Determine candidates for rigid divisors (d > W).
#      If any rigid divisor has no candidate, skip L.
#      If any two rigid divisors have the same candidate, skip L.
#   2. Collect the set of used values from rigid divisors.
#   3. For flexible divisors (d <= W), generate their candidate multiples in [L, L+W].
#   4. We need to pick one candidate for each flexible divisor such that:
#      - It is not in the set of used values.
#      - All picked values are distinct.
#      - All values are within [L, L+W] (by construction).
#   5. This is a small assignment problem (4 flexible divisors, few candidates each).
#      We can solve it with backtracking or simple permutation check.

def _solve_for_W(W):
    """
    Check if there exists a window of width W with distinct multiples.
    Returns (True, result_list) if found, else (False, None).
    """
    # Split divisors into rigid and flexible
    rigid_divs = [d for d in _DIVS if d > W]
    flex_divs = [d for d in _DIVS if d <= W]
    
    # Map divisor to index in _DIVS for reconstruction
    div_to_idx = {d: i for i, d in enumerate(_DIVS)}
    
    for L in range(_LCM):
        # 1. Handle rigid divisors
        used = {}  # value -> divisor index
        rigid_vals = {}
        valid = True
        
        for d in rigid_divs:
            # Find multiple of d in [L, L+W]
            rem = L % d
            if rem == 0:
                m = L
            else:
                m = L + (d - rem)
            
            if m > L + W:
                valid = False
                break
            
            if m in used:
                valid = False
                break
            
            used[m] = div_to_idx[d]
            rigid_vals[div_to_idx[d]] = m
        
        if not valid:
            continue
        
        # 2. Handle flexible divisors
        # Generate candidates for each flexible divisor
        flex_candidates = {}
        for d in flex_divs:
            idx = div_to_idx[d]
            cands = []
            rem = L % d
            if rem == 0:
                m = L
            else:
                m = L + (d - rem)
            
            while m <= L + W:
                if m not in used:
                    cands.append(m)
                m += d
            if not cands:
                valid = False
                break
            flex_candidates[idx] = cands
        
        if not valid:
            continue
        
        # 3. Backtracking for flexible divisors
        # We have 4 flexible divisors (usually).
        # We need to assign distinct values from their candidate lists.
        
        # Sort flex divs by number of candidates (most constrained first)
        flex_items = sorted(flex_candidates.items(), key=lambda x: len(x[1]))
        
        def backtrack(i, used_so_far, assignment):
            if i == len(flex_items):
                return True
            
            idx, cands = flex_items[i]
            for val in cands:
                if val not in used_so_far:
                    used_so_far.add(val)
                    assignment[idx] = val
                    if backtrack(i + 1, used_so_far, assignment):
                        return True
                    del assignment[idx]
                    used_so_far.remove(val)
            return False
        
        assignment = dict(rigid_vals)
        used_so_far = set(used.keys())
        
        if backtrack(0, used_so_far, assignment):
            # Reconstruct result list in order of _DIVS
            result = [0] * _N
            for idx, val in assignment.items():
                result[idx] = val
            
            # Check max value constraint
            if max(result) > _MAX_VAL:
                continue
            
            return (True, result)
    
    return (False, None)

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill
    
    best_result = None
    best_W = None
    
    # We know a solution exists with W <= 33 (seed).
    # Let's try W from 1 to 33.
    # If we find one, it's optimal because we start from 1.
    
    for W in range(1, 34):
        if time.time() > deadline:
            break
        
        ok, res = _solve_for_W(W)
        if ok:
            best_result = res
            best_W = W
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback to seed if no better found (should not happen)
    return [int(d) for d in _DIVS]

# EVOLVE-BLOCK-END
