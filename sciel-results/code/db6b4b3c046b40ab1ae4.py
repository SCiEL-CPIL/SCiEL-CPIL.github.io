"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    target = Fraction(5, 31)
    
    # Initial greedy solution to have a valid fallback
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # Search for solutions with k terms (k=2 to 6)
    # We search for d1 < d2 < ... < dk
    # For a fixed k, we iterate d1, d2, ..., d_{k-1} and calculate dk.
    
    def search_k_terms(k):
        nonlocal best_solution, best_max
        
        # Recursive helper to find d1..d_{k-1}
        def backtrack(depth, current_denoms, remaining_frac):
            nonlocal best_solution, best_max
            
            if time.time() > deadline:
                return
            
            # Pruning: Check if it's possible to represent remaining_frac
            # with (k - depth) terms, all > current_denoms[-1] (or 0 if depth 0)
            
            min_prev = current_denoms[-1] if current_denoms else 0
            terms_left = k - depth
            
            if terms_left == 1:
                # The last denominator is determined
                # remaining_frac = 1 / dk  => dk = 1 / remaining_frac
                if remaining_frac.numerator != 1:
                    return
                dk = remaining_frac.denominator
                if dk <= min_prev:
                    return
                # Check if dk is an integer (it is by construction if num==1)
                # Check if distinct (dk > min_prev ensures distinctness from previous)
                # Check if within limits
                if dk > 10000000:
                    return
                
                candidate = current_denoms + [dk]
                curr_max = candidate[-1] # since sorted
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = candidate
                return
            
            # Pruning bounds for next denominator d_next
            # We need: sum(1/d_i for i=depth+1..k) = remaining_frac
            # Since d_i are increasing:
            # 1. d_next > 1 / remaining_frac  (actually d_next >= floor(1/rem) + 1)
            # 2. terms_left * (1/d_next) > remaining_frac  => d_next < terms_left / remaining_frac
            
            min_d = max(min_prev + 1, remaining_frac.denominator // remaining_frac.numerator + 1)
            max_d = (terms_left * remaining_frac.denominator) // remaining_frac.numerator
            
            # Additional pruning:
            # The maximum possible sum with terms_left terms starting from min_d is:
            # terms_left / min_d. If this is < remaining_frac, no solution.
            if Fraction(terms_left, min_d) < remaining_frac:
                return
            
            # The minimum possible sum with terms_left terms starting from max_d (roughly)
            # Actually, if we pick d_next = max_d, the remaining terms must be > max_d.
            # The minimum sum they can provide is (terms_left - 1) / (max_d + 1) ... no, that's not tight enough.
            # Better: The remaining fraction after picking d_next is R' = remaining_frac - 1/d_next.
            # R' must be representable by (terms_left - 1) terms > d_next.
            # Max sum of (terms_left - 1) terms > d_next is (terms_left - 1) / (d_next + 1).
            # So we need R' <= (terms_left - 1) / (d_next + 1).
            
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return
                
                # Optimization: Skip if d is not a divisor of something? No, general case.
                # But we can check if (remaining_frac - 1/d) is positive.
                rem_next = remaining_frac - Fraction(1, d)
                if rem_next <= 0:
                    continue
                
                # Check if rem_next can be represented by (terms_left - 1) terms > d
                # Max sum of (terms_left - 1) terms with denominators > d is:
                # (terms_left - 1) / (d + 1)
                if rem_next > Fraction(terms_left - 1, d + 1):
                    continue
                
                # Min sum of (terms_left - 1) terms with denominators > d is:
                # This is tricky, but if rem_next is very small, it might be impossible?
                # Actually, we can always make the sum arbitrarily small by increasing denominators.
                # So the main constraint is the upper bound.
                
                current_denoms.append(d)
                backtrack(depth + 1, current_denoms, rem_next)
                current_denoms.pop()
        
        # Start backtracking for k terms
        backtrack(0, [], target)

    # Try k=2, 3, 4, 5, 6
    for k in range(2, 7):
        if time.time() > deadline:
            break
        search_k_terms(k)
        
        # If we found a solution with k terms, can we improve with k+1?
        # Yes, more terms might allow smaller max denominator.
        # But if we run out of time, we stop.
        
    return best_solution
# EVOLVE-BLOCK-END
