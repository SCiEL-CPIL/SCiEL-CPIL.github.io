import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # LCM of divisors
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    lcm_all = 30030
    
    # We search for the minimum span W.
    # The window [L, L+W] must contain distinct multiples.
    # Due to periodicity, we only need to check L in [1, lcm_all].
    
    # Helper to find candidates for a divisor d in range [L, L+W]
    def get_candidates(d, L, W):
        k_start = (L + d - 1) // d
        k_end = (L + W) // d
        if k_start > k_end:
            return []
        return [k * d for k in range(k_start, k_end + 1)]

    # Backtracking solver for a specific window
    def solve_window(L, W):
        # Generate candidates for all divisors
        candidates = []
        for d in divisors:
            c = get_candidates(d, L, W)
            if not c:
                return None
            candidates.append(c)
        
        # Sort divisors and candidates by number of options (MRV heuristic)
        # This helps prune the search tree significantly
        indexed_candidates = list(enumerate(candidates))
        indexed_candidates.sort(key=lambda x: len(x[1]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            orig_idx, opts = indexed_candidates[idx]
            for val in opts:
                if val not in used:
                    used.add(val)
                    assignment[orig_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[orig_idx] = 0
            return False

        if backtrack(0):
            return assignment
        return None

    # Iterate over possible span widths W
    # Minimum span for 10 distinct integers is 9.
    # We check W from 9 upwards.
    for W in range(9, 100):  # 100 is a safe upper bound, likely much smaller
        if time.time() > deadline:
            break
            
        # We slide the window L from 1 to lcm_all
        # Note: The window [L, L+W] must be within valid range.
        # Since we are in the first period, L+W <= 30030 + 100 < 1000000, so always valid.
        
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
                
            result = solve_window(L, W)
            if result is not None:
                # Verify distinctness and constraints just in case
                if len(set(result)) == n:
                    for i, val in enumerate(result):
                        if val < 1 or val > max_val or val % divisors[i] != 0:
                            # Should not happen, but handle gracefully
                            return list(divisors)
                    return list(result)
    
    # Fallback: Return divisors themselves (valid, distinct, but span is large)
    return list(divisors)
