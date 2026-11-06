"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We search for the minimum number of terms k
    # Based on analysis, k must be at least 6, likely 9 or 10.
    # We iterate k from 5 upwards.
    
    def dfs(remaining, min_d, terms_left, current):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining == 0:
                return current
            return None
            
        if terms_left == 1:
            # remaining must be 1/d
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d:
                    return current + [d]
            return None

        # Bounds for d
        # 1/d < remaining => d > 1/remaining
        # (terms_left - 1)/d > remaining => d < (terms_left - 1) / remaining
        
        # d must be integer > min_d
        # d_min = max(min_d, floor(1/remaining) + 1)
        # d_max = floor((terms_left - 1) / remaining)
        
        # Using Fraction
        # 1/remaining is Fraction(1, remaining)
        # d > 1/remaining => d >= floor(1/remaining) + 1 if 1/remaining is not integer, else 1/remaining + 1
        # Actually, strict inequality: 1/d < remaining => d > 1/remaining.
        
        lower_bound_val = Fraction(1, 1) / remaining
        # d must be > lower_bound_val
        d_min = int(lower_bound_val) + 1
        if d_min < min_d:
            d_min = min_d
            
        upper_bound_val = Fraction(terms_left - 1, 1) / remaining
        d_max = int(upper_bound_val)
        
        if d_min > d_max:
            return None
            
        # Iterate d
        for d in range(d_min, d_max + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break # Since d increases, 1/d decreases, new_remaining increases? No.
                # Wait. d increases => 1/d decreases => remaining - 1/d increases.
                # So if new_remaining < 0, it will get less negative? No.
                # If 1/d > remaining, then new_remaining < 0.
                # As d increases, 1/d decreases. So eventually 1/d < remaining.
                # So we should start d from d_min which ensures 1/d < remaining.
                # So new_remaining should be positive.
                
            res = dfs(new_remaining, d + 1, terms_left - 1, current + [d])
            if res is not None:
                return res
                
        return None

    # Try k from 5 to 15
    for k in range(5, 16):
        target = Fraction(1, 1)
        # Start with min denominator 5
        # We can optimize by fixing the first denominator 'a'
        # But DFS handles it.
        
        # Pruning: The first denominator a must satisfy a < k (from previous analysis a < k is necessary for existence)
        # So if k=5, a<5 is impossible since a>=5. So k=5 will fail quickly.
        
        result = dfs(target, 5, k, [])
        if result is not None:
            return result
            
    # Fallback: return a known valid solution if search fails or times out
    # This should not happen if k is small enough
    raise RuntimeError("No solution found within time limit")
# EVOLVE-BLOCK-END
