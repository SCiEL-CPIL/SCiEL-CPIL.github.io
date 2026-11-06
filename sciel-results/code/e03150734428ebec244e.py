"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_egyptian(target, min_denom, max_terms, deadline, used):
    """Recursively find Egyptian fraction for target with denominators >= min_denom."""
    if target == 0:
        return []
    if max_terms <= 0:
        return None
    if time.time() > deadline:
        return None
    
    # Target must be positive
    if target <= 0:
        return None
    
    # The denominator must satisfy 1/d <= target, so d >= 1/target
    # Also d >= min_denom
    # And we need enough remaining terms to make up the rest
    # For k remaining terms with min denominator d, max sum is 1/d + 1/(d+1) + ... 
    # which is bounded by k/d roughly
    
    lo = max(min_denom, -(-1 // target.numerator * target.denominator)) if target.numerator else min_denom
    # Actually: 1/d <= target => d >= 1/target
    # target = p/q, so d >= q/p
    import math
    lo = max(min_denom, math.ceil(target.denominator / target.numerator))
    
    # Upper bound on d: we need target - 1/d >= 0, and the remaining (max_terms-1) terms
    # with denominators >= d+1 must sum to at least target - 1/d
    # Max possible sum of (max_terms-1) terms with denominators >= d+1 is roughly (max_terms-1)/(d+1)
    # So we need target - 1/d <= (max_terms-1)/(d+1)
    # This gives an upper bound on d
    
    # Also, for the remaining terms to work, we need target - 1/d > 0 or == 0
    # If target - 1/d == 0, done
    # If target - 1/d < 0, invalid
    
    # Let's just iterate d from lo upward, with a reasonable cap
    # Cap: if d is too large, 1/d is too small to make progress
    # With max_terms terms left, the largest fraction we can still add is 1/lo
    # So we need target <= max_terms / lo roughly
    # If target > max_terms / lo, no solution from here
    
    if max_terms > 0 and target > Fraction(max_terms, lo):
        return None
    
    # Try denominators from lo upward
    # Limit search to avoid infinite loop
    hi = lo + 10000  # reasonable search window
    for d in range(lo, hi + 1):
        if d in used:
            continue
        frac = Fraction(1, d)
        if frac > target:
            continue
        new_target = target - frac
        if new_target == 0:
            return [d]
        if new_target < 0:
            continue
        # Check if new_target can be made with remaining terms
        if max_terms - 1 > 0 and new_target > Fraction(max_terms - 1, d + 1):
            continue
        result = _greedy_egyptian(new_target, d + 1, max_terms - 1, deadline, used | {d})
        if result is not None:
            return [d] + result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second deadline, 10s safety margin
    
    # Try to find minimal number of terms
    # Start with small k and increase
    best_result = None
    
    for k in range(2, 10):
        if time.time() > deadline:
            break
        result = _greedy_egyptian(Fraction(1, 1), 5, k, deadline, set())
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                best_result = result
                break
    
    if best_result is None:
        # Fallback: use the original DP approach
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
        raise RuntimeError("no representation found")
    
    return best_result
# EVOLVE-BLOCK-END
