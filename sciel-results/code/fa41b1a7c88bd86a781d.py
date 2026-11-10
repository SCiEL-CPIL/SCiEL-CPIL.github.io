from math import gcd
from fractions import Fraction

def construct():
    # Find minimum number of distinct denominators >= 5 summing to 1
    # Try n = 5, 6, 7, 8, ... until we find a solution
    
    for n in range(5, 13):
        result = _search(n)
        if result is not None:
            return result
    raise RuntimeError("no solution found")

def _search(n):
    # Find n distinct denominators >= 5 with sum of reciprocals = 1
    # Use backtracking with tight bounds
    
    target = Fraction(1, 1)
    
    def backtrack(start, remaining, chosen, remaining_terms):
        if remaining_terms == 0:
            return chosen if remaining == 0 else None
        
        if remaining <= 0:
            return None
            
        # Pruning: Max possible sum with remaining_terms terms starting from 'start'
        # If this max sum is less than 'remaining', we can't reach it.
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible += Fraction(1, start + i)
        if max_possible < remaining:
            return None
            
        # Determine the range for the next denominator d
        # 1. 1/d < remaining => d > 1/remaining
        # 2. d >= start
        lower_d = max(start, (1 // remaining.numerator) * remaining.denominator + 1) 
        # Actually, 1/d < remaining => d > 1/remaining.
        # d >= floor(1/remaining) + 1
        
        # Upper bound for d:
        # The remaining sum after picking d is remaining - 1/d.
        # This must be <= max sum of (remaining_terms-1) terms with denominators > d.
        # Max sum is sum_{i=1}^{remaining_terms-1} 1/(d+i).
        # So remaining - 1/d <= sum_{i=1}^{remaining_terms-1} 1/(d+i).
        # This inequality defines the upper bound for d.
        # As d increases, LHS increases (towards remaining) and RHS decreases.
        # So there is a cutoff.
        
        upper_d = 1000000 # Safe initial upper bound
        
        for d in range(lower_d, upper_d + 1):
            # Check if 1/d < remaining
            if Fraction(1, d) >= remaining:
                if remaining_terms == 1:
                    if Fraction(1, d) == remaining:
                        return chosen + [d]
                continue
                
            new_remaining = remaining - Fraction(1, d)
            
            # Check if we can complete with remaining_terms-1 terms
            # Max sum of remaining terms
            max_sum = Fraction(0, 1)
            for i in range(1, remaining_terms):
                max_sum += Fraction(1, d + i)
                
            if max_sum < new_remaining:
                # Even with the smallest possible denominators (d+1, d+2...),
                # we can't reach new_remaining.
                # Since d is increasing, max_sum will only get smaller.
                # So we can break.
                break
                
            # Recurse
            result = backtrack(d + 1, new_remaining, chosen + [d], remaining_terms - 1)
            if result is not None:
                return result
                
        return None
    
    # Start with smallest denominator 5
    return backtrack(5, target, [], n)
