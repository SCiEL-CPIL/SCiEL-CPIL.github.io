import math
import time
from fractions import Fraction

def find_solution_backtrack(k, deadline):
    """
    Find k distinct denominators >= 5 summing to 1.
    Returns list of denominators or None.
    """
    target = Fraction(1, 1)
    
    def max_sum_from(start_d, terms):
        """Calculate max sum of 'terms' distinct reciprocals starting from start_d."""
        s = Fraction(0, 1)
        for i in range(terms):
            s += Fraction(1, start_d + i)
        return s

    def dfs(start_d, remaining_sum, terms_left, current_solution):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining_sum == 0:
                return current_solution
            return None
            
        if terms_left == 1:
            # 1/d = remaining_sum => d = 1/remaining_sum
            if remaining_sum == 0:
                return None
            d = int(1 / remaining_sum)
            # Check if it's an integer and >= start_d
            if remaining_sum * d == 1 and d >= start_d and d <= 10000000:
                return current_solution + [d]
            return None

        # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
        # Since d is integer, d >= ceil(1/remaining_sum)
        # Also d >= start_d
        min_d = max(start_d, math.ceil(1 / float(remaining_sum)))
        
        # Upper bound for d:
        # We need the remaining terms to sum to remaining_sum - 1/d
        # The max sum of (terms_left-1) terms starting from d+1 must be >= remaining_sum - 1/d
        # Also, 1/d + min_sum_from(d+1, terms_left-1) <= remaining_sum
        # min_sum_from(d+1, terms_left-1) = sum of 1/(d+1+i) for i in range(terms_left-1)
        # This is a loose bound but helps prune
        
        # Better upper bound: if we pick d, then remaining_sum - 1/d must be achievable
        # with terms_left-1 terms each >= 1/(d+1), 1/(d+2), ...
        # Max possible sum of remaining terms = sum_{i=0}^{terms_left-2} 1/(d+1+i)
        # This must be >= remaining_sum - 1/d
        
        # For the upper bound on d, we need:
        # 1/d + max_sum_from(d+1, terms_left-1) >= remaining_sum
        # As d increases, 1/d decreases and max_sum_from decreases, so this becomes harder.
        # We can binary search or just iterate with a reasonable upper bound.
        
        # A simple upper bound: d <= terms_left / remaining_sum (from 1/d <= remaining_sum/terms_left)
        # But this is loose. Let's use a tighter bound based on the constraint.
        
        max_d = int(float(terms_left) / float(remaining_sum)) + 10
        if max_d < min_d:
            return None
            
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
                
            next_remaining = remaining_sum - Fraction(1, d)
            if next_remaining < 0:
                # d too small, increase d
                continue
            
            if next_remaining == 0:
                if terms_left == 1:
                    return current_solution + [d]
                else:
                    continue
            
            # Check if it's possible to reach next_remaining with terms_left-1 terms starting from d+1
            if terms_left - 1 > 0:
                # Max sum of terms_left-1 terms starting from d+1
                max_possible = Fraction(0, 1)
                for i in range(terms_left - 1):
                    max_possible += Fraction(1, d + 1 + i)
                
                if max_possible < next_remaining:
                    # Even with smallest possible denominators, we can't reach the sum
                    # Increasing d will make max_possible even smaller
                    break
            
            # Min sum check: the smallest possible sum of terms_left-1 terms starting from d+1
            # is sum_{i=0}^{terms_left-2} 1/(d+1+i) ... wait, that's the max.
            # The min would be using the largest denominators, which is unbounded.
            # So no useful min bound here.
            
            # Recurse
            res = dfs(d + 1, next_remaining, terms_left - 1, current_solution + [d])
            if res is not None:
                return res
                
        return None

    # Start search
    # First denominator d1.
    for d1 in range(5, 30):
        if time.time() > deadline:
            return None
        next_remaining = 1 - Fraction(1, d1)
        if next_remaining <= 0:
            continue
        
        # Check max possible sum for remaining k-1 terms
        max_possible = Fraction(0, 1)
        for i in range(k - 1):
            max_possible += Fraction(1, d1 + 1 + i)
            
        if max_possible < next_remaining:
            continue
            
        res = dfs(d1 + 1, next_remaining, k - 1, [d1])
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    # 90s limit. Use 85s for safety.
    deadline = start_time + 85.0
    
    # Known solution k=10
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = 10
    
    # Try k=7 first (hardest, but if found, best score)
    if time.time() < deadline - 10.0:
        sol_7 = find_solution_backtrack(7, deadline)
        if sol_7 is not None:
            return sol_7
        
    # Try k=8
    if time.time() < deadline - 5.0:
        sol_8 = find_solution_backtrack(8, deadline)
        if sol_8 is not None:
            return sol_8
        
    # Try k=9
    if time.time() < deadline - 2.0:
        sol_9 = find_solution_backtrack(9, deadline)
        if sol_9 is not None:
            return sol_9
            
    # Fallback to k=10
    return best_sol

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
