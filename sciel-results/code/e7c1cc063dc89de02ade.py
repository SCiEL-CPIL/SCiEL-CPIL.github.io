"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill
    
    # Known baseline solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def find_solution(max_denom):
        """
        Find any solution for 5/31 using distinct denominators in [1, max_denom].
        Uses integer arithmetic and aggressive pruning.
        Returns list of denominators or None.
        """
        
        def dfs(num, den, min_d, path):
            if time.time() - start_time > time_limit:
                return None
            if num == 0:
                return path
            if num < 0:
                return None
            
            # Lower bound for next denominator d:
            # 1/d <= num/den  =>  d >= den/num
            # d must be integer, so d >= ceil(den/num)
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > max_denom:
                return None
            
            # Additional pruning: if the maximum possible sum of reciprocals
            # from lower_bound to max_denom is less than num/den, impossible
            # Approximate sum_{d=lower_bound}^{max_denom} 1/d
            # Using integral approximation: ln(max_denom/lower_bound) + 1/lower_bound
            if max_denom > lower_bound:
                # Quick upper bound check
                # Sum is at most (max_denom - lower_bound + 1) / lower_bound
                # But tighter: use harmonic approximation
                # For now, rely on the iterative check which is fast enough
                pass
            
            for d in range(lower_bound, max_denom + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Compute next remainder: num/den - 1/d = (num*d - den)/(den*d)
                next_num = num * d - den
                if next_num < 0:
                    # 1/d > num/den, try larger d
                    continue
                
                if next_num == 0:
                    return path + [d]
                
                next_den = den * d
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Pruning: Can we form next_num/next_den with denoms in [d+1, max_denom]?
                # Upper bound on sum of 1/j for j in [d+1, max_denom]
                # Approx: ln(max_denom/(d+1)) + 1/(d+1)
                # Quick check: if (max_denom - d) / (d + 1) < next_num/next_den, prune
                if d < max_denom:
                    upper_bound_sum = (max_denom - d) / (d + 1)
                    if upper_bound_sum < next_num / next_den:
                        break
                
                res = dfs(next_num, next_den, d + 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(5, 31, 1, [])
    
    # Strategy: Binary search on the maximum denominator bound
    # We know a solution exists for K=930
    # We want to find the minimum K such that a solution exists
    
    # First, try to find a solution with a tighter bound
    # Try some heuristic bounds first to narrow the range
    test_bounds = [100, 200, 300, 500, 700, 800, 900]
    feasible_bounds = []
    
    for K in test_bounds:
        if K >= best_max_denom:
            continue
        if time.time() - start_time > time_limit:
            break
        
        sol = find_solution(K)
        if sol is not None:
            feasible_bounds.append(K)
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = list(sol)
    
    # Now do binary search in the feasible range
    if feasible_bounds:
        search_low = feasible_bounds[0]
        search_high = best_max_denom - 1
        
        while search_low < search_high and time.time() - start_time < time_limit - 2.0:
            mid = (search_low + search_high) // 2
            sol = find_solution(mid)
            
            if sol is not None:
                actual_max = max(sol)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = list(sol)
                search_high = mid
            else:
                search_low = mid + 1
        
        # Try the final bound
        if time.time() - start_time < time_limit - 1.0:
            sol = find_solution(search_low)
            if sol is not None:
                actual_max = max(sol)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = list(sol)
    
    # Verify the solution
    from fractions import Fraction
    target = Fraction(5, 31)
    if best_solution is not None and len(best_solution) > 0 and len(best_solution) <= 256:
        total = sum(Fraction(1, d) for d in best_solution)
        if total == target and all(d >= 1 and d <= 10000000 for d in best_solution):
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback: return the known good solution
    return [10, 15, 930]
# EVOLVE-BLOCK-END
