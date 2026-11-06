"""Combined program: greedy Egyptian fraction with exact arithmetic and bounded DFS."""
import time
import math
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit

    def find_min_terms(target, min_den, max_terms, used):
        """
        Find the minimum number of terms (up to max_terms) to represent target
        as a sum of distinct unit fractions with denominators >= min_den.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None

        if target == 0:
            return []

        if max_terms <= 0:
            return None

        # If target is a unit fraction and its denominator is valid, return it
        if target.numerator == 1:
            d = target.denominator
            if d >= min_den and d not in used:
                return [d]
            return None

        # Lower bound on denominator: 1/d <= target => d >= 1/target
        lo = max(min_den, math.ceil(target.denominator / target.numerator))

        # Upper bound on denominator:
        # We need target - 1/d >= 0 => d >= 1/target (already handled)
        # Also, the remaining (max_terms-1) terms must be able to sum to target - 1/d.
        # The maximum sum of (max_terms-1) terms with denominators >= d+1 is roughly (max_terms-1)/(d+1).
        # So we need target - 1/d <= (max_terms-1)/(d+1)
        # target - 1/d <= (max_terms-1)/(d+1)
        # (target - 1/d)(d+1) <= max_terms - 1
        # target*d + target - 1 - 1/d <= max_terms - 1
        # This is getting complex. Let's use a simpler heuristic bound.
        # d cannot be too large, otherwise 1/d is too small and we can't reach the target
        # with the remaining terms.
        # Max possible sum of remaining terms is (max_terms-1) / (d+1)
        # So we need target - 1/d <= (max_terms-1)/(d+1)
        # Let's just iterate d from lo up to a reasonable limit.
        # A safe upper limit: if d is very large, 1/d is tiny.
        # Let's try up to lo + 50000 or until we find a solution.
        
        # Pruning: if the maximum possible sum of max_terms terms starting from min_den
        # is less than target, no solution.
        max_possible = sum(Fraction(1, min_den + i) for i in range(max_terms))
        if max_possible < target:
            return None

        # Iterate denominators
        # We try smaller denominators first (greedy)
        # But we also need to ensure that the remaining target can be formed.
        
        # Upper bound for d in this step:
        # We need target - 1/d >= 0 => d >= 1/target
        # And we need target - 1/d <= sum of 1/(d+1) + ... for (max_terms-1) terms.
        # The max sum of (max_terms-1) terms with denominators >= d+1 is (max_terms-1)/(d+1).
        # So target - 1/d <= (max_terms-1)/(d+1)
        # target*d + target - 1 - 1/d <= max_terms - 1
        # Let's just cap d to a reasonable value to avoid infinite loops.
        # If d > target.denominator * max_terms, 1/d is very small.
        
        d_max = lo + 100000  # Cap search window
        if d_max > 10000000:
            d_max = 10000000

        for d in range(lo, d_max + 1):
            if time.time() > deadline:
                return None
            
            if d in used:
                continue

            frac = Fraction(1, d)
            if frac > target:
                continue

            new_target = target - frac
            if new_target < 0:
                continue

            # Pruning: check if new_target can be formed with remaining terms
            if max_terms - 1 > 0:
                # Max sum of (max_terms-1) terms with denominators >= d+1
                max_rem = sum(Fraction(1, d + 1 + i) for i in range(max_terms - 1))
                if new_target > max_rem:
                    continue
                
                # Min sum of (max_terms-1) terms with denominators <= 10000000
                # This is tricky because denominators must be distinct and > d.
                # But min sum is close to (max_terms-1)/10000000.
                # If new_target < (max_terms-1)/10000000, no solution.
                min_rem = Fraction(max_terms - 1, 10000000)
                if new_target < min_rem:
                    continue

            result = find_min_terms(new_target, d + 1, max_terms - 1, used | {d})
            if result is not None:
                return [d] + result

        return None

    # Try to find solution with increasing number of terms
    for k in range(2, 8):  # Try up to 7 terms
        if time.time() > deadline:
            break
        result = find_min_terms(Fraction(1, 1), 5, k, set())
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result

    # Fallback: DP over divisors of 2520
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no solution found")
