# EVOLVE-BLOCK-START
import time
import math

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _is_valid_answer(ans, divisors):
    """Check if answer is valid: 10 distinct positive ints <= 1e6, each divisible by corresponding divisor."""
    if not isinstance(ans, (list, tuple)):
        return False
    if len(ans) != len(divisors):
        return False
    seen = set()
    for i, (val, div) in enumerate(zip(ans, divisors)):
        if not isinstance(val, int) or isinstance(val, bool):
            return False
        if val < 1 or val > P['maximum']:
            return False
        if val % div != 0:
            return False
        if val in seen:
            return False
        seen.add(val)
    return True

def _span(ans):
    return max(ans) - min(ans)

def _check_window(L, S, divisors):
    """
    Check if there is a valid assignment of distinct multiples for each divisor
    within the window [L, L+S].
    Returns the list of values if found, else None.
    """
    n = len(divisors)
    candidates = []
    for i, d in enumerate(divisors):
        k_min = (L + d - 1) // d
        k_max = (L + S) // d
        if k_min < 1:
            k_min = 1
        if k_min > k_max:
            return None
        vals = [k * d for k in range(k_min, k_max + 1)]
        candidates.append((i, d, vals))
    
    # Sort by number of candidates (MRV heuristic)
    candidates.sort(key=lambda x: len(x[2]))
    
    assigned = [None] * n
    used_values = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        orig_idx, d, vals = candidates[idx]
        for v in vals:
            if v not in used_values:
                used_values.add(v)
                assigned[orig_idx] = v
                if backtrack(idx + 1):
                    return True
                used_values.remove(v)
                assigned[orig_idx] = None
        return False
    
    if backtrack(0):
        return assigned[:]
    return None

def construct():
    """
    Find 10 distinct positive integers <= 1000000, aligned with divisors [6,10,14,15,21,22,26,33,35,39].
    Each representative must be divisible by its corresponding input.
    Minimize max(representatives) - min(representatives).
    """
    divisors = P['divisors']
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Fallback solution: the divisors themselves
    best_ans = divisors[:]
    best_span = _span(best_ans)
    
    # Binary search on Span S
    # We know S=33 is feasible (the divisors themselves).
    # We want to find the minimum S.
    lo = 1
    hi = best_span
    
    # We will scan L from 1 to L_MAX for each S.
    # If S is small, L_MAX can be smaller? No, keep it constant for simplicity.
    # 5000 is a safe limit. 
    L_MAX = 5000
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        found = False
        for L in range(1, L_MAX + 1):
            if time.time() > deadline:
                break
            ans = _check_window(L, mid, divisors)
            if ans is not None:
                if _is_valid_answer(ans, divisors):
                    return [int(v) for v in ans]
                found = True
                break
        
        if found:
            hi = mid - 1
        else:
            lo = mid + 1
            
    return [int(v) for v in best_ans]

# EVOLVE-BLOCK-END
