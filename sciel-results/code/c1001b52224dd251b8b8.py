"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Find Egyptian fraction expansion of 5/31 minimizing the largest denominator."""
    deadline = time.time() + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Lower bound: smallest possible denominator is ceil(31/5) = 7
    # The max denominator must be at least as large as the smallest denominator
    # Try increasing D (max allowed denominator) and check feasibility
    
    def search_bounded(remaining, min_den, max_den, used, depth=0):
        """Check if we can express 'remaining' as sum of unit fractions
        with all denominators in [min_den, max_den], not in used, distinct."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            return list(used)
        if remaining < 0:
            return None
        if depth >= 10:  # reasonable depth limit
            return None
        
        # Lower bound on next denominator: must satisfy 1/d <= remaining
        # so d >= 1/remaining
        if remaining > 0:
            lo = max(min_den, (1 + remaining.numerator - 1) // remaining.numerator)
            # Actually: 1/d <= remaining means d >= 1/remaining
            # 1/remaining = remaining.denominator / remaining.numerator
            lo = max(min_den, remaining.denominator // remaining.numerator)
            if lo < min_den:
                lo = min_den
            # Also need 1/lo <= remaining, i.e., lo >= 1/remaining
            # 1/remaining = den/num, so lo >= den/num
            lo = max(lo, remaining.denominator // remaining.numerator)
            # Ensure 1/lo <= remaining
            while lo < max_den and Fraction(1, lo) > remaining:
                lo += 1
        else:
            return None
        
        # Upper bound: we need enough remaining terms to sum to 'remaining'
        # With at most (max_den - lo + 1) terms available, but practically limit
        # Also, the next denominator can't be too large: if 1/d is too small,
        # we can't reach the target with remaining terms
        # Simple bound: d must satisfy that remaining - 1/d >= 0 and achievable
        hi = max_den
        
        # Pruning: if even using the largest possible denominators we can't reach target
        # The maximum sum we can achieve with denominators from d to max_den is bounded
        # But this is complex; let's just iterate
        
        for d in range(lo, hi + 1):
            if d in used:
                continue
            if Fraction(1, d) > remaining:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
            used.add(d)
            result = search_bounded(new_remaining, d + 1, max_den, used, depth + 1)
            if result is not None:
                return result
            used.remove(d)
        
        return None
    
    # Try increasing maximum denominator D
    # Lower bound for D: at least 7 (smallest possible denominator)
    # Upper bound: we know 248 works, so search up to 247 to find improvement
    # If nothing found, return the known solution
    
    best_solution = [8, 31, 248]
    
    for D in range(7, 248):
        if time.time() > deadline:
            break
        used = set()
        result = search_bounded(target, 7, D, used, 0)
        if result is not None:
            # Verify the solution
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(1 <= d <= 10000000 for d in result) and 1 <= len(result) <= 256:
                best_solution = sorted(result)
                break  # Found smallest D, stop
    
    return best_solution
# EVOLVE-BLOCK-END
