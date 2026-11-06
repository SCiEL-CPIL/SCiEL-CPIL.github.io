import time
from fractions import Fraction
import math
from itertools import combinations

def construct():
    target = Fraction(5, 31)
    
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    def dfs(rem, last_d, K_val, depth, max_depth):
        if rem == 0:
            return []
        if depth >= max_depth:
            return None
        
        d_min = last_d + 1
        if rem > 0:
            d_min_candidate = int(1 / rem) + 1
            if d_min_candidate > d_min:
                d_min = d_min_candidate
        
        if d_min > K_val:
            return None
            
        for d in range(d_min, K_val + 1):
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                # This should not happen if d_min is calculated correctly,
                # but if it does, larger d will make 1/d smaller, so new_rem will increase.
                # So we should NOT break. We should continue.
                # Wait.
                # If new_rem < 0, then 1/d > rem.
                # As d increases, 1/d decreases.
                # So for larger d, 1/d might be <= rem.
                # So we cannot break. We must continue.
                pass
            
            if new_rem == 0:
                return [d]
            
            result = dfs(new_rem, d, K_val, depth + 1, max_depth)
            if result is not None:
                return [d] + result
        
        return None

    # Search for minimal K
    for K in range(7, 1737):
        if time.time() - start_time > time_limit:
            break
        
        sol = dfs(target, 0, K, 0, 10)
        if sol is not None:
            if verify(sol):
                return sol
                
    return best_solution
