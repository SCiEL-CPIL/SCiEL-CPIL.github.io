"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _search_min_terms():
    """Find minimum number of distinct denominators >= 5 summing to 1."""
    deadline = time.time() + 80
    best = None

    def dfs(remaining, min_den, current, max_depth):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) >= max_depth:
            return
        if remaining <= 0:
            if remaining == 0:
                if best is None or len(current) < len(best):
                    best = list(current)
            return
        # Each remaining term is at most 1/min_den
        # Need at least remaining * min_den terms (loose bound)
        # But more tightly: the max sum of k terms with den >= min_den is bounded
        # Try denominators
        # d must satisfy 1/d <= remaining => d >= 1/remaining
        # And we need to be able to fill remaining - 1/d with (max_depth - len(current) - 1) terms
        lo = min_den
        # Upper bound: 1/d < remaining => d > 1/remaining
        hi = int(1 / remaining) + 1
        hi = min(hi, 10000000)
        for d in range(lo, hi + 1):
            if time.time() > deadline:
                return
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if best is None or len(current) + 1 < len(best):
                    best = list(current) + [d]
                return
            # Pruning: check feasibility
            # Max possible sum from (max_depth - len(current) - 1) terms with den >= d+1
            # is at most (max_depth - len(current) - 1) / (d + 1)
            remaining_terms = max_depth - len(current) - 1
            if remaining_terms <= 0:
                return
            if new_rem > remaining_terms / (d + 1):
                # Can't reach this much with remaining terms
                continue
            dfs(new_rem, d + 1, current + [d], max_depth)

    # 4 terms impossible (max sum < 1), so start from 5
    for max_depth in range(5, 9):
        dfs(Fraction(1, 1), 5, [], max_depth)
        if best is not None:
            break
        if time.time() > deadline:
            break

    return best

def construct():
    result = _search_min_terms()
    if result is None:
        # Fallback: seed program's DP approach
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
        raise RuntimeError("no baseline representation")
    return result
# EVOLVE-BLOCK-END
