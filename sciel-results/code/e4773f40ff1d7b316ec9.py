"""
Strategy: Sliding window search over LCM period with rigid/flexible divisor decomposition.
For W < 21, divisors > W are rigid (0 or 1 multiple in window).
Iterate W from 0 upward, L from 0 to LCM-1, find first valid assignment.
Optimized with precomputation and fast feasibility checks.
"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

# Precompute LCM
def _lcm(a, b):
    return a * b // gcd(a, b)
_LCM = reduce(_lcm, P['divisors'])  # 30030

# Precompute divisor indices: rigid vs flexible depends on W
# Divisors sorted for reference
_DIVS = P['divisors']
_N = len(_DIVS)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (evaluator kills at 90s)
    
    best_W = 33  # Seed span
    best_solution = None
    
    # We iterate W from 0 upwards to find the minimum span.
    # We stop as soon as we find a valid L for a given W.
    for W in range(0, best_W + 1):
        if time.time() > deadline:
            break
        found = False
        for L in range(0, _LCM):
            if time.time() > deadline:
                break
            sol = _find_assignment(L, W)
            if sol is not None:
                best_W = W
                best_solution = sol
                found = True
                break
        if found:
            break
            
    if best_solution is None:
        # Fallback to seed
        return list(_DIVS)
        
    return best_solution

def _find_assignment(L, W):
    """
    Returns a list of 10 distinct integers in [L, L+W] divisible by corresponding divisors,
    or None if not possible.
    """
    divisors = _DIVS
    n = _N
    hi = L + W
    used = [0] * (hi - L + 1)  # used[v-L] = True if value v is taken
    assignments = [0] * n
    flex_idx = []
    
    # 1. Handle rigid divisors (d > W): at most one multiple in [L, hi]
    for i in range(n):
        d = divisors[i]
        if d > W:
            # Find the unique multiple of d in [L, hi]
            k = (L + d - 1) // d
            val = k * d
            if val > hi:
                return None
            idx = val - L
            if used[idx]:
                return None
            used[idx] = 1
            assignments[i] = val
        else:
            flex_idx.append(i)
    
    # 2. Handle flexible divisors via backtracking (MRV order)
    # Build candidate lists
    cand_lists = []
    for i in flex_idx:
        d = divisors[i]
        cands = []
        k = (L + d - 1) // d
        val = k * d
        while val <= hi:
            if not used[val - L]:
                cands.append(val)
            val += d
        cand_lists.append(cands)
    
    # Sort by candidate count (MRV)
    order = sorted(range(len(flex_idx)), key=lambda x: len(cand_lists[x]))
    
    # Backtrack
    def _backtrack(pos):
        if pos == len(order):
            return True
        fi = order[pos]
        orig_i = flex_idx[fi]
        for val in cand_lists[fi]:
            idx = val - L
            if not used[idx]:
                used[idx] = 1
                assignments[orig_i] = val
                if _backtrack(pos + 1):
                    return True
                used[idx] = 0
                assignments[orig_i] = 0
        return False
    
    if _backtrack(0):
        return assignments
    return None

# EVOLVE-BLOCK-END
