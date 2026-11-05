"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    def dfs(start, remaining, target):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return [] if target == 0 else None
        
        # Pruning 1: Max possible sum
        # Sum of 1/(start+i) for i in range(remaining)
        # If this < target, no solution
        # Optimization: Use approximation or simple sum for small 'remaining'
        max_sum = sum(Fraction(1, start + i) for i in range(remaining))
        if max_sum < target:
            return None
        
        # Determine range for current denominator d
        # 1/d <= target => d >= 1/target
        # d >= ceil(target.denominator / target.numerator)
        min_d = max(start, math.ceil(target.denominator / target.numerator))
        
        # 1/d >= target / remaining => d <= remaining / target
        # d <= floor(remaining * target.numerator / target.denominator)
        max_d = min(10000000, math.floor(remaining * target.numerator / target.denominator))
        
        if min_d > max_d:
            return None
            
        for d in range(min_d, max_d + 1):
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            res = dfs(d + 1, remaining - 1, new_target)
            if res is not None:
                return [d] + res
        return None

    # Try k=8, then k=9, then k=10
    for k in [8, 9, 10]:
        result = dfs(5, k, Fraction(1, 1))
        if result is not None:
            # Verify
            if len(result) == len(set(result)) and all(5 <= d <= 10000000 for d in result):
                if sum(Fraction(1, d) for d in result) == 1:
                    return result

    # Fallback: Greedy (will produce many terms, but valid)
    # This is just a safety net, unlikely to be reached
    denoms = []
    target = Fraction(1, 1)
    used = set()
    while target > 0 and len(denoms) < 256:
        min_d = max(5, math.ceil(1.0 / float(target)))
        if min_d < 5:
            min_d = 5
        d = min_d
        while d in used or Fraction(1, d) > target:
            d += 1
        if d > 10000000:
            break
        denoms.append(d)
        used.add(d)
        target -= Fraction(1, d)
    if target == 0 and len(denoms) <= 256:
        return denoms
        
    # Absolute fallback
    return [5, 6, 7, 10, 14, 21, 30, 42, 70, 140]
# EVOLVE-BLOCK-END
