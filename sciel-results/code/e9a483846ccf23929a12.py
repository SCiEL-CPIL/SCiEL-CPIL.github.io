"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    target = Fraction(5, 31)
    
    # Strategy: Systematic search for Egyptian fraction representations
    # We want to minimize the LARGEST denominator.
    # 
    # Key insight: For a 3-term representation 1/a + 1/b + 1/c = 5/31,
    # we can fix a and b, then solve for c:
    # 1/c = 5/31 - 1/a - 1/b = (5ab - 31a - 31b) / (31ab)
    # c = 31ab / (5ab - 31a - 31b)
    # For c to be a positive integer, (5ab - 31a - 31b) must divide 31ab.
    # Also need c > b >= a >= 7 (since 1/6 > 5/31, a must be >= 7)
    #
    # We'll search for the minimum possible max(a,b,c) by iterating
    # possible values of the largest denominator in increasing order.
    
    # First, let's find a good baseline solution
    # Known: 5/31 = 1/8 + 1/31 + 1/248 (max = 248)
    # Let's verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31 ✓
    
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Search for 3-term solutions with smaller max denominator
    # For 3 terms: 1/a + 1/b + 1/c = 5/31 with a <= b <= c
    # We need c to be minimized, so we want c < best_max
    
    # Iterate c from 7 upward (c is the largest, so c >= b >= a >= 7)
    # For each c, check if there exist a <= b <= c such that 1/a + 1/b = 5/31 - 1/c
    
    # Actually, let's iterate over possible max values M from 7 upward
    # and check if a valid subset of {7,...,M} sums to 5/31
    
    # More efficient: iterate over the largest term c from 7 to best_max-1
    # For each c, we need 1/a + 1/b = 5/31 - 1/c where a <= b <= c
    
    # 1/a + 1/b = (5c - 31) / (31c)
    # Let R = (5c - 31) / (31c)
    # 1/a + 1/b = R means (a+b)/(ab) = R, so R*ab = a+b
    # ab*R - a - b = 0
    # a(b*R - 1) = b
    # a = b / (b*R - 1) = b / (b*(5c-31)/(31c) - 1) = 31c*b / (b*(5c-31) - 31c)
    
    # For a to be integer: (b*(5c-31) - 31c) must divide 31c*b
    # Let D = b*(5c-31) - 31c
    # Need D > 0 and D | 31c*b
    
    # Also a <= b, so 31c*b / D <= b, meaning 31c <= D = b*(5c-31) - 31c
    # 62c <= b*(5c-31), so b >= 62c/(5c-31)
    
    # And b <= c
    
    # Let's search c from 7 to best_max - 1
    for c in range(7, best_max):
        if time.time() > deadline:
            break
        
        # We need 1/a + 1/b = 5/31 - 1/c
        # The RHS must be positive: 5/31 > 1/c, i.e., c > 31/5 = 6.2, so c >= 7 ✓
        
        # R = (5c - 31) / (31c)
        # We need a <= b <= c, a >= 7
        # b >= ceil(62c/(5c-31)) and b <= c
        
        num = 5 * c - 31
        if num <= 0:
            continue
        
        # b must satisfy: b >= 62c/(5c-31) and b <= c
        # Also b >= a >= 7
        
        b_min = max(7, (62 * c + num - 1) // num)  # ceil(62c/num)
        b_max = c
        
        if b_min > b_max:
            continue
        
        # For each b in [b_min, b_max], check if a is integer
        # a = 31c*b / (b*num - 31c)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            
            D = b * num - 31 * c
            if D <= 0:
                continue
            
            # Check if D divides 31c*b
            if (31 * c * b) % D != 0:
                continue
            
            a = (31 * c * b) // D
            
            if a < 7 or a > b:
                continue
            
            # Verify
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s == target:
                # Found a solution with max = c < best_max
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
                    break
    
    # Now try 4-term and more-term solutions for even smaller max
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31 with a <= b <= c <= d
    # We want d < best_max
    # 
    # This is more complex. Let's use a DFS approach for small M values.
    
    # Check if we can do better with more terms
    # For M < best_max, check if any subset of {7,...,M} sums to 5/31
    # Use memoization/DP with Fraction sums
    
    # Actually, let's try a different approach for small M:
    # For M from 7 to best_max - 1, use DFS with strong pruning
    
    # Precompute suffix sums for efficiency
    def can_form(remaining, start, M, used):
        """DFS to check if we can form 'remaining' using denominators from start to M"""
        if remaining == 0:
            return []
        if start > M:
            return None
        if time.time() > deadline:
            return None
        
        # Pruning: if sum of all available < remaining, impossible
        # Sum of 1/k for k in [start, M]
        # For large ranges, approximate; for small, compute exactly
        if M - start < 50:
            avail_sum = sum(Fraction(1, k) for k in range(start, M + 1))
        else:
            # Approximate
            import math
            avail_sum = math.log(M) - math.log(start - 1) + 1e-15
        
        if avail_sum < remaining:
            return None
        
        # If 1/start > remaining, skip start
        if Fraction(1, start) > remaining:
            return can_form(remaining, start + 1, M, used)
        
        # Try not using start
        result = can_form(remaining, start + 1, M, used)
        if result is not None:
            return result
        
        # Try using start
        if start not in used:
            new_used = used | {start}
            result = can_form(remaining - Fraction(1, start), start + 1, M, new_used)
            if result is not None:
                return [start] + result
        
        return None
    
    # Search for solutions with max < best_max using more terms
    # Start from M = best_max - 1 down to 7
    # But this might be slow. Let's be strategic.
    
    # Actually, the 3-term search above should have found the optimal 3-term solution.
    # For more terms, the max denominator could potentially be smaller.
    # E.g., with 4 terms, we might get max < 248 (or whatever best_max is now).
    
    # Let's do a limited DFS search for M < best_max
    # Focus on M values that are likely to work
    
    # For efficiency, only try M values where the total sum is close to target
    # sum_{k=7}^{M} 1/k should be >= 5/31
    
    # sum_{k=7}^{M} 1/k ≈ ln(M/6) + 0.577 - (1+1/2+1/3+1/4+1/5+1/6)
    # = ln(M/6) + 0.577 - 2.45
    # = ln(M/6) - 1.873
    # Set >= 0.1613: ln(M/6) >= 2.034, M/6 >= 7.65, M >= 46
    
    # So M should be at least ~46 for the total sum to be sufficient
    # But with selective terms, we might need less
    
    # Let's try DFS for M from 7 to min(best_max - 1, 100) with time limit
    for M in range(7, min(best_max, 100)):
        if time.time() > deadline:
            break
        
        # Quick check: is total sum >= target?
        total = sum(Fraction(1, k) for k in range(7, M + 1))
        if total < target:
            continue
        
        # Also, the minimum sum of any valid subset... 
        # Just try DFS with time limit
        # Use a set to track used denominators
        
        # For efficiency, use a different approach:
        # Iterate over possible first terms
        
        found = False
        # Try all subsets of size 2, 3, 4 from {7,...,M}
        # This is combinatorially expensive for large M, but M < 100
        
        # For M <= 50, we can try combinations
        if M - 7 + 1 <= 40:  # at most 40 elements
            # Try combinations of size 2
            for a, b in combinations(range(7, M + 1), 2):
                if time.time() > deadline:
                    break
                s = Fraction(1, a) + Fraction(1, b)
                if s > target:
                    continue
                rem = target - s
                if rem == 0:
                    best_max = max(a, b)
                    best_solution = [a, b]
                    found = True
                    break
                # Check if rem can be formed by one more term
                # 1/c = rem, c = 1/rem
                if rem > 0:
                    c_val = 1 / rem
                    if c_val == int(c_val) and 7 <= int(c_val) <= M and int(c_val) not in (a, b):
                        best_max = max(a, b, int(c_val))
                        best_solution = [a, b, int(c_val)]
                        found = True
                        break
            if found:
                break
        
        if time.time() > deadline:
            break
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
