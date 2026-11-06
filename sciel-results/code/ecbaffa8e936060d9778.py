import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Known 10-term solution
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = 10

    def max_sum(m, start_d):
        """Calculate max sum of m distinct unit fractions with denominators > start_d."""
        s = 0.0
        for i in range(m):
            s += 1.0 / (start_d + 1 + i)
        return s

    def dfs(k, remaining_sum, min_denom, current_sol):
        """
        Try to find a solution with exactly k terms.
        remaining_sum: Fraction
        min_denom: minimum allowed denominator for the next term
        current_sol: list of chosen denominators
        Returns list of denominators if found, else None.
        """
        if time.time() > deadline:
            return None

        if k == 0:
            if remaining_sum == 0:
                return current_sol
            else:
                return None

        if k == 1:
            # 1/d = remaining_sum => d = 1/remaining_sum
            if remaining_sum.numerator == 1:
                d = remaining_sum.denominator
                if d > min_denom and d <= 10000000:
                    return current_sol + [d]
            return None

        # Pruning: Max possible sum with k terms starting from min_denom+1
        if max_sum(k, min_denom) < float(remaining_sum):
            return None

        # Determine range for next denominator d
        # 1/d <= remaining_sum => d >= ceil(1/remaining_sum)
        # Also d > min_denom
        min_d = max(min_denom + 1, math.ceil(1.0 / float(remaining_sum)))
        
        # Upper bound for d:
        # remaining_sum - 1/d <= max_sum(k-1, d)
        # 1/d >= remaining_sum - max_sum(k-1, d)
        # This is a bit circular. Let's just iterate d from min_d upwards.
        # How high can d go?
        # If d is very large, 1/d is small, remaining_sum - 1/d is close to remaining_sum.
        # But max_sum(k-1, d) decreases as d increases.
        # So there is a cutoff.
        
        # Let's estimate the max d.
        # max_sum(k-1, d) approx (k-1)/d.
        # remaining_sum - 1/d <= (k-1)/d
        # remaining_sum <= k/d
        # d <= k / remaining_sum
        # This is a loose upper bound. A tighter one:
        # remaining_sum - 1/d <= 1/(d+1) + ... + 1/(d+k-1)
        
        # We will iterate d. To avoid infinite loops, we can stop if 1/d < remaining_sum / k? No.
        # We can stop if max_sum(k-1, d) < remaining_sum - 1/d.
        
        for d in range(min_d, 10000001):
            if time.time() > deadline:
                return None
            
            # Check if 1/d is too small
            if Fraction(1, d) > remaining_sum:
                break # d is too small, but we started from min_d, so this shouldn't happen unless float error
            
            rem = remaining_sum - Fraction(1, d)
            if rem < 0:
                break
            
            # Pruning: Can rem be represented by k-1 terms with denom > d?
            if max_sum(k-1, d) < float(rem):
                break # d is too large, rem is too big for the remaining terms?
                # Wait. If max_sum(k-1, d) < rem, it means the remaining terms are too small to sum to rem.
                # So we need a smaller d to make 1/d larger, thus rem smaller.
                # But we are increasing d. So if this condition is met, increasing d further will make 1/d smaller, rem larger, and max_sum(k-1, d) smaller.
                # So it will never work. Break.
            
            # Additional pruning:
            # The smallest possible sum of k-1 terms with denom > d is 0 (approaching).
            # But we need exact match.
            
            res = dfs(k-1, rem, d, current_sol + [d])
            if res:
                return res
        
        return None

    # Try k=7, 8, 9
    for k in [7, 8, 9]:
        if time.time() > deadline:
            break
        sol = dfs(k, Fraction(1, 1), 4, [])
        if sol:
            # Verify
            total = sum(Fraction(1, d) for d in sol)
            if total == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                best_k = k
                best_sol = sol
                break

    return best_sol
