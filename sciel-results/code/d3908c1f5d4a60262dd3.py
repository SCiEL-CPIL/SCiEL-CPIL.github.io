from math import gcd
from fractions import Fraction
import sys

def construct():
    # Find minimum number of distinct denominators >= 5 summing to 1
    # Known: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/42 = 1? Let me verify.
    # Actually, let's just search for the minimum.
    
    for n in range(5, 13):
        result = _search(n)
        if result is not None:
            return result
    raise RuntimeError("no solution found")

def _search(n):
    # Find n distinct denominators >= 5 with sum of reciprocals = 1
    # Use backtracking with integer arithmetic for speed
    
    target_num = 1
    target_den = 1
    
    def backtrack(start, num, den, remaining_terms):
        if remaining_terms == 0:
            return [] if num == 0 else None
        
        if remaining_terms == 1:
            # Need 1/d = num/den, so d = den/num
            if den % num == 0:
                d = den // num
                if d >= start and d >= 5 and d <= 10000000:
                    return [d]
            return None
        
        # Pruning: max possible sum with remaining_terms terms starting from 'start'
        # The largest reciprocals are 1/start, 1/(start+1), ...
        # If their sum < num/den, we can't reach the target
        max_possible = 0.0
        for i in range(remaining_terms):
            max_possible += 1.0 / (start + i)
        if max_possible < num / den - 1e-15:
            return None
        
        # Lower bound for d: 1/d < num/den => d > den/num
        # Also d >= start
        lower_d = max(start, den // num + 1)
        
        # Upper bound for d: we need to be able to represent (num/den - 1/d) with (remaining_terms-1) terms
        # each with denominator > d. The maximum sum of those terms is sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # So we need num/den - 1/d <= sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # This gives an upper bound on d.
        
        # Let's iterate d from lower_d upward, with pruning
        # As d increases, num/den - 1/d increases (approaches num/den) and the max sum decreases
        # So there's a point where we can break
        
        upper_d = 10000000  # cap
        
        for d in range(lower_d, upper_d + 1):
            # 1/d must be < num/den (guaranteed by lower_d)
            # New remainder: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            if new_num <= 0:
                continue
            new_den = den * d
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Check if (remaining_terms-1) terms with denominators > d can sum to new_num/new_den
            # Max sum is sum_{i=1}^{remaining_terms-1} 1/(d+i)
            max_rem = 0.0
            for i in range(1, remaining_terms):
                max_rem += 1.0 / (d + i)
            
            if new_num / new_den > max_rem + 1e-12:
                # Can't reach the remainder, and as d increases this gap only widens
                break
            
            # Recurse
            result = backtrack(d + 1, new_num, new_den, remaining_terms - 1)
            if result is not None:
                return [d] + result
        
        return None
    
    # Start with smallest denominator 5
    result = backtrack(5, target_num, target_den, n)
    return result
