import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    target = Fraction(5, 31)
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0.0
        return math.log(n) + gamma + 1.0/(2.0*n) - 1.0/(12.0*n*n)
    
    def can_complete(remainder, min_denom, max_denom):
        if remainder == 0:
            return True
        if remainder < 0:
            return False
        if max_denom <= min_denom:
            return False
        
        inv_rem = 1 / remainder
        d_min_req = math.ceil(inv_rem)
        d_start = max(min_denom + 1, d_min_req)
        
        if d_start > max_denom:
            return False
        
        max_sum_tight = harmonic_approx(max_denom) - harmonic_approx(d_start - 1)
        if max_sum_tight < float(remainder) - 1e-12:
            return False
        
        return True

    def dfs(remainder, min_denom, max_denom, current_set):
        if time.time() - start_time > time_limit:
            return None
        
        if remainder == 0:
            return current_set
        
        if remainder < 0:
            return None
        
        if not can_complete(remainder, min_denom, max_denom):
            return None
        
        d_min = max(min_denom + 1, math.ceil(1 / remainder))
        d_max = max_denom
        
        if d_min > d_max:
            return None
        
        for d in range(d_min, d_max + 1):
            if time.time() - start_time > time_limit:
                return None
            
            next_rem = remainder - Fraction(1, d)
            
            if next_rem < 0:
                continue
            
            if next_rem == 0:
                return current_set + [d]
            
            result = dfs(next_rem, d, max_denom, current_set + [d])
            if result is not None:
                return result
        
        return None

    # Try greedy descent: start from a small bound and increase
    # First, try to find a solution with a small max denominator
    
    # Try bounds from small to large, but with a smarter approach
    # Start with the known best and try to improve
    
    # Try a range of bounds, focusing on smaller ones first
    # Use a more aggressive search for small bounds
    
    bounds_to_try = []
    # Try small bounds first
    for k in range(20, best_max_den + 1, 10):
        bounds_to_try.append(k)
    # Also try specific values near the expected optimum
    for k in range(100, 200):
        if k not in bounds_to_try:
            bounds_to_try.append(k)
    bounds_to_try.sort()
    
    for K in bounds_to_try:
        if time.time() - start_time > time_limit:
            break
        if K >= best_max_den:
            continue
        
        solution = dfs(target, 0, K, [])
        
        if solution is not None:
            actual_max = max(solution)
            if actual_max < best_max_den:
                best_max_den = actual_max
                best_solution = solution
                # Continue searching for even smaller
    
    # Final: try to find exact minimum by binary search around the best found
    if best_max_den < 1736:
        low = max(1, best_max_den - 50)
        high = best_max_den - 1
        
        while low <= high and time.time() - start_time < time_limit:
            mid = (low + high) // 2
            solution = dfs(target, 0, mid, [])
            
            if solution is not None:
                actual_max = max(solution)
                if actual_max < best_max_den:
                    best_max_den = actual_max
                    best_solution = solution
                high = mid - 1
            else:
                if time.time() - start_time > time_limit:
                    break
                low = mid + 1
    
    return best_solution
