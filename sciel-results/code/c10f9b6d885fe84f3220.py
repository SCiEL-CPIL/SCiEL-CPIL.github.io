"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 85
    
    def search(k, min_d, remaining, used):
        """
        Find k distinct denominators >= min_d summing to `remaining`.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining > 0:
                # remaining = 1/d => d = 1/remaining
                # d must be integer
                num, den = remaining.numerator, remaining.denominator
                if num == 1:
                    d = den
                    if d >= min_d and d >= 5 and d not in used and d <= 10000000:
                        return [d]
            return None
        
        # Pruning: Max possible sum with k terms starting from min_d
        # is sum(1/(min_d + i) for i in range(k))
        # If this max sum < remaining, no solution.
        max_possible = sum(Fraction(1, min_d + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Determine search range for the first denominator d
        # 1. d >= min_d
        # 2. 1/d < remaining => d > 1/remaining => d >= floor(1/remaining) + 1
        lower_bound = max(min_d, remaining.denominator // remaining.numerator + 1)
        
        # 3. Upper bound: 
        # We need remaining - 1/d <= sum_{i=1}^{k-1} 1/(d+i)
        # Let's iterate d and break if this condition fails.
        # The condition is: remaining - 1/d <= max_sum_rest(d)
        # where max_sum_rest(d) = sum_{i=1}^{k-1} 1/(d+i)
        
        for d in range(lower_bound, 10000001):
            if time.time() > deadline:
                return None
            
            if d in used:
                continue
            
            frac = Fraction(1, d)
            if frac >= remaining:
                # Since d is increasing, frac is decreasing.
                # If frac >= remaining at the start, we should have started higher?
                # No, lower_bound ensures 1/d < remaining.
                # So this check is mostly for safety or if lower_bound was loose.
                # Actually if 1/d >= remaining, we can't use it because rest must be positive.
                # But lower_bound is derived from 1/d < remaining.
                pass
            
            new_remaining = remaining - frac
            
            # Check if new_remaining can be formed by k-1 terms > d
            if k - 1 > 0:
                # Max sum of k-1 terms starting from d+1
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
                if max_rest < new_remaining:
                    # If the maximum possible sum of the remaining terms is less than 
                    # what we need, then d is too large (1/d was too small).
                    # Since d is increasing, 1/d is decreasing, new_remaining is increasing.
                    # max_rest is decreasing.
                    # So if it fails now, it will fail for all larger d.
                    break
            
            result = search(k - 1, d + 1, new_remaining, used | {d})
            if result is not None:
                return [d] + result
        
        return None

    # We know k must be at least 7.
    # Try k = 7, 8, 9...
    for k in range(7, 12):
        if time.time() > deadline:
            break
        result = search(k, 5, Fraction(1, 1), set())
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == k and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback if search fails (should not happen for k=7 or 8)
    # Use the DP approach from inspirations as a safety net, though it might return non-minimal k.
    # However, the task asks to "minimize their count".
    # If the search fails, it's better to return a valid solution than crash.
    # But given the math, k=7 is very likely solvable.
    
    # A known solution for k=7?
    # Let's rely on the search.
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
