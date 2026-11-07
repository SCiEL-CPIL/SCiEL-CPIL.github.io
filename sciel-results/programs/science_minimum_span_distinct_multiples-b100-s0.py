import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)  # LCM of all divisors
    
    best_diff = float('inf')
    best_solution = None
    
    # Precompute: for each divisor, precompute multiples in [1, 2*period]
    # Since L in [1, period] and W in [1, period], values in [1, 2*period]
    max_search = 2 * period
    
    # For faster candidate lookup
    # candidates_for_divisor[d] = list of multiples of d in [1, max_search]
    candidates_cache = {}
    for d in divisors:
        cands = []
        x = d
        while x <= max_search:
            cands.append(x)
            x += d
        candidates_cache[d] = cands
    
    def check_assignment(L, W):
        R = L + W
        # For each divisor, get candidate multiples in [L, R]
        candidates = []
        for d in divisors:
            cands_all = candidates_cache[d]
            # Binary search for first index where value >= L
            lo_idx, hi_idx = 0, len(cands_all)
            while lo_idx < hi_idx:
                mid_idx = (lo_idx + hi_idx) // 2
                if cands_all[mid_idx] < L:
                    lo_idx = mid_idx + 1
                else:
                    hi_idx = mid_idx
            start_idx = lo_idx
            # Find end index where value <= R
            end_idx = start_idx
            while end_idx < len(cands_all) and cands_all[end_idx] <= R:
                end_idx += 1
            candidates.append(cands_all[start_idx:end_idx])
        
        # Check if any divisor has no candidates
        for i in range(n):
            if len(candidates[i]) == 0:
                return None
        
        # Sort by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return [0] * n
            
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    result = backtrack(idx + 1)
                    if result is not None:
                        result[i] = val
                        return result
                    used.remove(val)
            return None
        
        result = backtrack(0)
        if result is not None:
            return result
        return None
    
    # Iterate over all possible starting positions L in one period
    for L in range(1, period + 1):
        if time.time() > deadline:
            break
        
        # Binary search on W (window width)
        lo = 1
        hi = period
        best_w_for_L = None
        best_sol_for_L = None
        
        while lo <= hi:
            mid = (lo + hi) // 2
            sol = check_assignment(L, mid)
            if sol is not None:
                best_w_for_L = mid
                best_sol_for_L = sol
                hi = mid - 1
            else:
                lo = mid + 1
        
        if best_sol_for_L is not None:
            actual_vals = best_sol_for_L
            diff = max(actual_vals) - min(actual_vals)
            if diff < best_diff:
                best_diff = diff
                best_solution = actual_vals[:].copy()
                if best_diff == 1:
                    break  # Can't do better than 1
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
