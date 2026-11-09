"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 80
    
    # Known valid 7-term solution (verified: sum of reciprocals = 1)
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 != 1
    # Let's use a verified one found via search
    # After computation: [5, 6, 7, 8, 9, 10, 11] doesn't work
    # Valid solution: [5, 6, 7, 8, 9, 12, 18] -> check
    # 1/5+1/6+1/7+1/8+1/9+1/12+1/18 = (36+30+40+45+40+30+20)/360 = 241/360 != 1
    
    # Let me use a different approach: find one quickly
    def find_solution(k_max=8):
        """Find a solution with at most k_max terms using efficient search."""
        best = None
        
        for k in range(7, k_max + 1):
            if time.time() > deadline:
                break
            result = _search(k, 5, Fraction(1, 1), frozenset())
            if result is not None:
                return result
        return best
    
    def _search(k, min_d, remaining, used):
        """Efficient backtracking search for k denominators."""
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            num, den = remaining.numerator, remaining.denominator
            if num == 1:
                d = den
                if d >= min_d and d >= 5 and d not in used and d <= 10000000:
                    return [d]
            return None
        
        # Pruning: max possible sum with k terms starting from min_d
        max_possible = sum(Fraction(1, min_d + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Lower bound: 1/d < remaining => d > 1/remaining
        lower_bound = max(min_d, remaining.denominator // remaining.numerator + 1)
        
        # Upper bound: we need remaining - 1/d <= (k-1)/(d+1) approximately
        # So 1/d >= remaining - (k-1)/(d+1)
        # For safety, iterate but break early
        upper_bound = min(10000000, lower_bound + 100000)  # Limit search range
        
        for d in range(lower_bound, upper_bound + 1):
            if time.time() > deadline:
                return None
            if d in used:
                continue
            
            frac = Fraction(1, d)
            if frac >= remaining:
                break  # d too small, but lower_bound should prevent this
            
            new_remaining = remaining - frac
            if new_remaining <= 0:
                continue
            
            # Check if remaining can be formed by k-1 terms > d
            if k - 1 > 0:
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
                if max_rest < new_remaining:
                    # d is too large, break
                    break
            
            result = _search(k - 1, d + 1, new_remaining, used | frozenset([d]))
            if result is not None:
                return [d] + result
        
        return None
    
    # Try to find optimal solution
    result = find_solution(k_max=8)
    
    if result is not None:
        # Verify
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
            return result
    
    # Fallback: use a known valid solution
    # This is a verified 7-term Egyptian fraction for 1 with denominators >= 5
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 is NOT valid
    # Let's use: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12
    # = 0.2 + 0.1667 + 0.1429 + 0.125 + 0.1111 + 0.1 + 0.0833 = 0.929, not 1
    
    # I need to find a valid one. Let me use a different known decomposition.
    # Using the fact that 1 = 1/2 + 1/3 + 1/6, and splitting terms:
    # 1/2 = 1/5 + 1/10
    # 1/3 = 1/6 + 1/18  (since 1/3 = 1/6 + 1/18? No, 1/6+1/18 = 4/18 = 2/9 != 1/3)
    # 1/3 = 1/7 + 1/21 (since 1/7+1/21 = 4/21 != 1/3)
    # 1/3 = 1/8 + 1/12 (since 1/8+1/12 = 5/24 != 1/3)
    # 1/3 = 1/9 + 1/9, not distinct
    # 1/3 = 1/10 + 1/15 (since 1/10+1/15 = 3/30 = 1/10, no)
    # Actually 1/10 + 1/15 = 3/30 + 2/30 = 5/30 = 1/6, not 1/3
    
    # Let me just return a larger valid solution as fallback
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + ...
    # This is getting complex. Let me use the DP approach as fallback.
    
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            sol = reached[target]
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    # Last resort: construct a valid solution using splitting
    # Start with 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + remainder
    # This is complex. Just raise an error if we can't find one.
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
