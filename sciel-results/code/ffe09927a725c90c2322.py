"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        """Approximate the n-th harmonic number."""
        if n <= 0:
            return 0
        if n == 1:
            return 1.0
        return math.log(n) + gamma + 1.0/(2.0*n) - 1.0/(12.0*n*n)
    
    def exact_harmonic_sum(n1, n2):
        """
        Compute exact harmonic sum H(n2) - H(n1-1) = sum(1/d for d in n1..n2)
        using Fraction for exactness, but with early termination if sum exceeds threshold.
        """
        if n1 > n2:
            return Fraction(0, 1)
        # For small ranges, compute directly
        if n2 - n1 < 200:
            s = Fraction(0, 1)
            for d in range(n1, n2 + 1):
                s += Fraction(1, d)
            return s
        # For larger ranges, use approximation with a safety factor
        # We'll use the approximation but add a margin
        approx = harmonic_approx(n2) - harmonic_approx(n1 - 1)
        # Return as a slightly inflated Fraction to be safe
        return Fraction(int(approx * 1000000), 1000000)
    
    def can_complete(remainder, min_denom, max_denom):
        """
        Check if it's possible to sum to 'remainder' using distinct denominators 
        in [min_denom, max_denom].
        """
        if remainder == 0:
            return True
        if remainder < 0:
            return False
        
        if min_denom > max_denom:
            return False
        
        # Max possible sum
        max_sum = exact_harmonic_sum(min_denom, max_denom)
        
        if max_sum < remainder:
            return False
        
        # Minimum number of terms needed
        min_terms_needed = math.ceil(float(remainder) * min_denom)
        available_terms = max_denom - min_denom + 1
        if min_terms_needed > available_terms:
            return False
        
        return True
    
    def find_solution_bounded(max_denom, min_denom_start=7):
        """
        Find a solution with all denominators <= max_denom.
        Returns list of denominators or None.
        """
        if time.time() - start_time > time_limit:
            return None
        
        target_rem = Fraction(5, 31)
        used = set()
        solution = []
        
        def dfs(remainder, min_d, depth):
            nonlocal best_solution, best_max_denom
            
            if time.time() - start_time > time_limit:
                return None
                
            if remainder == 0:
                return list(solution)
                
            if remainder < 0:
                return None
                
            if not can_complete(remainder, min_d, max_denom):
                return None
                
            # Lower bound for next denominator
            if remainder > 0:
                lower_bound = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
            else:
                lower_bound = min_d
            
            if lower_bound < min_d:
                lower_bound = min_d
                
            if lower_bound > max_denom:
                return None
                
            if depth > 256:
                return None
            
            # Try denominators from lower_bound to max_denom
            for d in range(lower_bound, max_denom + 1):
                if time.time() - start_time > time_limit:
                    return None
                    
                if d in used:
                    continue
                    
                next_rem = remainder - Fraction(1, d)
                
                if next_rem < 0:
                    continue
                
                if next_rem == 0:
                    solution.append(d)
                    result = list(solution)
                    solution.pop()
                    return result
                
                used.add(d)
                solution.append(d)
                result = dfs(next_rem, d + 1, depth + 1)
                if result is not None:
                    solution.pop()
                    used.remove(d)
                    return result
                solution.pop()
                used.remove(d)
            
            return None
        
        return dfs(target_rem, min_denom_start, 0)
    
    # Strategy: Binary search on the maximum denominator bound
    low = 7  # Minimum possible (1/d < 5/31 means d > 31/5 = 6.2)
    high = best_max_denom
    
    # First, try to find a better solution with a tighter bound
    # Do a few iterations of binary search
    
    # To speed up, we can try some heuristic bounds first
    test_bounds = [100, 200, 300, 500, 800, 1200, 1500]
    feasible_bounds = []
    
    for K in test_bounds:
        if K > high:
            continue
        if time.time() - start_time > time_limit:
            break
        
        result = find_solution_bounded(K)
        
        if result is not None:
            feasible_bounds.append(K)
            if len(result) > 0:
                actual_max = max(result)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = result
            if best_max_denom <= K:
                break
    
    # Now do binary search in the feasible range
    if feasible_bounds:
        search_low = feasible_bounds[0]
        search_high = high
        
        while search_low < search_high and time.time() - start_time < time_limit:
            mid = (search_low + search_high) // 2
            result = find_solution_bounded(mid)
            
            if result is not None:
                search_high = mid
                if len(result) > 0:
                    actual_max = max(result)
                    if actual_max < best_max_denom:
                        best_max_denom = actual_max
                        best_solution = result
            else:
                search_low = mid + 1
        
        # Try the final bound
        result = find_solution_bounded(search_low)
        if result is not None and len(result) > 0:
            actual_max = max(result)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = result
    
    # If we still have the original solution or found a better one, return it
    if best_solution is not None and len(best_solution) > 0 and len(best_solution) <= 256:
        total = sum(Fraction(1, d) for d in best_solution)
        if total == target and all(d >= 1 and d <= 10000000 for d in best_solution):
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback: return the known good solution
    return [8, 28, 1736]
# EVOLVE-BLOCK-END
