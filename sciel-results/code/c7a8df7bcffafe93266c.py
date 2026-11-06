import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        # Verify sum equals 5/31 using integer arithmetic
        # sum(1/d) == 5/31  <=>  31 * sum(1/d) == 5
        # Use LCM-based verification or just compute with fractions
        from fractions import Fraction
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def find_solution_with_limit(limit):
        """
        Finds any solution for 5/31 using distinct denominators in [1, limit].
        Returns the list of denominators or None.
        Uses integer arithmetic: state is (num, den) representing num/den.
        """
        # DFS with pruning
        # State: (num, den, min_d, current_set)
        # We want sum of 1/d_i = num/den
        
        def dfs(num, den, min_d, current_set):
            if time.time() - start_time > time_limit:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound for next denominator d:
            # 1/d <= num/den  =>  d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Upper bound pruning: 
            # Even if we use the smallest possible denominators from lower_bound onward,
            # can we reach num/den?
            # Max possible sum with remaining terms is bounded by harmonic series tail.
            # For simplicity, use a tighter bound:
            # If 1/lower_bound > num/den, we can't use it (but lower_bound ensures 1/d <= num/den)
            # Actually, we need: sum of 1/d for d from lower_bound to limit >= num/den
            # This is always true if we have enough terms, but let's add a quick check.
            # 
            # A better prune: the maximum number of terms we can add is (limit - min_d + 1),
            # but we don't want to enumerate. Instead, use:
            # If num/den > sum_{d=lower_bound}^{limit} 1/d, then impossible.
            # But computing this sum is expensive. Skip for now.
            
            # Iterate possible denominators
            # Heuristic: Try larger d first to keep denominators small in the set?
            # No, we want to find ANY solution. Try smaller d first (greedy-like) 
            # because it reduces the remainder faster.
            
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # Check: 1/d <= num/den  <=>  den <= num * d
                if den > num * d:
                    # This d is too small (1/d > num/den)
                    # But d >= lower_bound = ceil(den/num), so this should not happen
                    # due to integer division. Skip to be safe.
                    continue
                
                next_num = num * d - den
                next_den = den * d
                
                # Reduce fraction
                g = gcd(next_num, next_den)
                if g > 1:
                    next_num //= g
                    next_den //= g
                
                # Recurse
                res = dfs(next_num, next_den, d + 1, current_set + [d])
                if res is not None:
                    return res
                    
            return None
        
        # Start with remainder 5/31, min_denom 1
        return dfs(5, 31, 1, [])

    # Binary search on the maximum denominator bound
    # If no solution exists with max denom <= K, then no solution exists with max denom <= K-1
    # So we can binary search for the smallest K where a solution exists.
    
    # First, check if our baseline is valid
    if not verify(best_solution):
        # Fallback: greedy construction
        from fractions import Fraction
        r = Fraction(5, 31)
        solution = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > 10000000:
                break
            solution.append(d)
            r -= Fraction(1, d)
        if solution and verify(solution):
            best_solution = solution
            best_max_denom = max(solution)
        else:
            return [8, 28, 1736]  # Should not happen
    
    # Binary search: find the smallest K such that a solution exists with all denoms <= K
    # We know a solution exists with K = best_max_denom
    # Check if a solution exists with K = best_max_denom - 1
    
    low = 1
    high = best_max_denom
    
    # First, verify that high is feasible (it should be, since best_solution works)
    # But our DFS might not find it. Let's just start the search.
    
    # Actually, let's do a linear descent from best_max_denom - 1 downward,
    # since the answer is likely close to best_max_denom.
    
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        sol = find_solution_with_limit(current_limit)
        
        if sol is not None:
            # Found a solution with max <= current_limit
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            
            # Try even smaller limit
            current_limit = actual_max - 1
        else:
            # No solution found for this limit.
            # By monotonicity, no solution exists for any smaller limit.
            # So best_solution is optimal.
            break
        
        # Safety: if we've gone too low, stop
        if current_limit < 7:  # 1/7 is the smallest possible first term for 5/31
            break

    # Final verification
    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
