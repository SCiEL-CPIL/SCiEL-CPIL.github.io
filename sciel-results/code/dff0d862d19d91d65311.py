import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Calculate minimum k such that sum of reciprocals of k smallest allowed denominators >= 1
    # Allowed denominators start at 5.
    def max_sum_for_k(k):
        s = Fraction(0)
        for i in range(k):
            s += Fraction(1, 5 + i)
        return s

    min_k = 5
    while max_sum_for_k(min_k) < 1:
        min_k += 1
        
    # Search for solutions with increasing k
    for k in range(min_k, 257):
        solution = []
        
        def backtrack(i, d_last, S):
            if time.time() > deadline:
                return False
            
            if i == k:
                if S == 1:
                    return True
                else:
                    return False
            
            remaining_terms = k - i
            
            # Pruning 1: Max possible sum with remaining terms
            # Smallest possible denominators are d_last+1, d_last+2, ...
            max_add = Fraction(0)
            for j in range(remaining_terms):
                max_add += Fraction(1, d_last + 1 + j)
            
            if S + max_add < 1:
                return False
            
            # Pruning 2: Current sum already exceeds 1
            if S > 1:
                return False
                
            # Determine bounds for next denominator d
            # Lower bound:
            # 1. d > d_last
            # 2. 1/d <= 1 - S  => d >= 1 / (1 - S)
            lower_d = d_last + 1
            if S < 1:
                val = Fraction(1, 1) / (1 - S)
                # ceil(val)
                lb = val.numerator // val.denominator
                if val.numerator % val.denominator != 0:
                    lb += 1
                lower_d = max(lower_d, lb)
            
            # Upper bound:
            # We need 1/d + sum(1/(d+1+j) for j in range(remaining_terms-1)) >= 1 - S
            # Approx: (remaining_terms)/d >= 1 - S  => d <= remaining_terms / (1 - S)
            upper_d = 10000000
            if S < 1:
                val = Fraction(remaining_terms, 1) / (1 - S)
                ub = val.numerator // val.denominator
                # Add 1 to be safe
                upper_d = min(upper_d, ub + 1)
            
            # Iterate d
            for d in range(lower_d, upper_d + 1):
                new_S = S + Fraction(1, d)
                
                # If new_S > 1, since d is increasing, 1/d is decreasing, 
                # but we already ensured lower_d such that 1/d <= 1-S.
                # So new_S <= 1 should hold.
                
                rem_terms = remaining_terms - 1
                if rem_terms == 0:
                    if new_S == 1:
                        solution.append(d)
                        return True
                    else:
                        continue
                
                # Check if we can still reach 1
                max_rest = Fraction(0)
                for j in range(rem_terms):
                    max_rest += Fraction(1, d + 1 + j)
                    
                if new_S + max_rest < 1:
                    # As d increases, new_S decreases and max_rest decreases.
                    # So we can break.
                    break
                
                solution.append(d)
                if backtrack(i + 1, d, new_S):
                    return True
                solution.pop()
            
            return False
        
        if backtrack(0, 4, Fraction(0)):
            return solution
        
    # Fallback: Return a known valid solution if search fails (should not happen)
    # But we must return a valid list.
    # If we didn't find one, we raise an error.
    raise RuntimeError("No solution found within time limit")
