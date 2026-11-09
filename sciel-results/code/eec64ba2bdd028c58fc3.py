"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    # We know that 3, 4, 5, 6, 7 terms are impossible for distinct denominators >= 5.
    # 8 terms are possible. We search for a solution with exactly 8 terms.
    
    target_depth = 8
    target_sum = Fraction(1, 1)
    
    def dfs(rem, min_d, depth, path):
        # rem is the remaining sum to be formed by 'target_depth - depth' terms
        # min_d is the minimum allowed denominator (must be > last chosen)
        # depth is the current number of terms chosen
        
        if depth == target_depth:
            if rem == 0:
                return path
            return None
        
        # Pruning: Check if it's possible to reach 'rem' with remaining slots
        slots_left = target_depth - depth
        # The largest possible sum we can add is using the smallest possible denominators
        max_possible_sum = Fraction(0, 1)
        d = min_d
        for _ in range(slots_left):
            max_possible_sum += Fraction(1, d)
            d += 1
        
        if max_possible_sum < rem:
            return None
            
        # The smallest possible sum we can add (using very large denominators) is effectively 0,
        # but we must ensure 1/d <= rem for the next step.
        # So the next denominator d_next must satisfy 1/d_next <= rem => d_next >= 1/rem.
        # Also d_next >= min_d.
        
        start_d = max(min_d, math.ceil(1 / float(rem)))
        
        # Upper bound for d_next:
        # We need to leave enough sum for the remaining (slots_left - 1) terms.
        # The minimum sum for (slots_left - 1) terms starting from d_next + 1 is small,
        # so we can just iterate. To be safe, we can bound d_next such that
        # 1/d_next >= rem - max_sum(remaining_slots).
        # But simple iteration from start_d upwards is fine as d won't be huge.
        
        for d in range(start_d, 10000): # 10000 is a safe upper bound for small depth
            if rem < Fraction(1, d):
                break
            if rem == Fraction(1, d) and slots_left == 1:
                return path + [d]
            
            # Recurse
            new_rem = rem - Fraction(1, d)
            result = dfs(new_rem, d + 1, depth + 1, path + [d])
            if result:
                return result
                
        return None

    solution = dfs(target_sum, 5, 0, [])
    
    if solution:
        return solution
    else:
        # Fallback to a known valid solution if DFS fails (should not happen)
        # Example: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/15.75... no.
        # Just return a valid set found by previous logic or hardcode one.
        # Based on analysis, a solution exists.
        raise RuntimeError("No 8-term solution found")

# EVOLVE-BLOCK-END
