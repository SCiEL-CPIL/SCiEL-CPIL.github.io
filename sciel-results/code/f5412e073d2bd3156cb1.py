"""Improved program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # First, try greedy with various starting denominators to get a baseline
    best_max = float('inf')
    best_solution = None
    
    def greedy_complete(r, min_den, max_den, max_terms=255):
        """Complete Egyptian fraction greedily, respecting bounds. Return list or None."""
        values = []
        used = set()
        for _ in range(max_terms):
            if r == 0:
                return values
            if time.time() > deadline:
                return None
            # Greedy: smallest d >= min_den such that 1/d <= r and d <= max_den
            # d must satisfy: r <= 1/d => d <= 1/r
            # and d >= min_den, d >= ceil(1/r) for 1/d <= r
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            if d in used:
                return None
            used.add(d)
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None
    
    # Try greedy completions from various starting denominators
    for first_d in range(7, 50):
        if time.time() > deadline:
            break
        r = target - Fraction(1, first_d)
        if r < 0:
            continue
        # Try greedy completion with max_den = best_max - 1 to see if we can improve
        for max_den in [best_max - 1, 500, 1000, 2000]:
            if max_den < first_d + 1:
                continue
            sol = greedy_complete(r, first_d + 1, max_den)
            if sol is not None:
                full_sol = [first_d] + sol
                if len(full_sol) <= 256:
                    m = max(full_sol)
                    if m < best_max:
                        best_max = m
                        best_solution = full_sol
    
    # Now do a bounded DFS search to find solutions with max denominator < best_max
    # This is more thorough but slower
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        """DFS to find a representation with all denominators in [min_den, max_den]."""
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current and max(current) < best_max:
                best_max = max(current)
                best_solution = list(current)
            return True
        if depth >= max_depth:
            return False
        if r <= 0:
            return False
        
        # Bound: we need sum of remaining terms = r
        # Each term <= 1/min_den, so we need at least ceil(r * min_den) terms
        # Also, the largest possible contribution from remaining terms is bounded
        
        # Try denominators from min_den to max_den
        # Pruning: 1/d <= r required, so d >= ceil(1/r)
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try to find a solution with max denominator < best_max
    # Search with decreasing max_den bounds
    for search_max in range(min(best_max - 1, 5000), 100, -100):
        if time.time() > deadline:
            break
        if dfs(target, 1, search_max, [], 0):
            break
    
    # Final check: if we found something, return it; otherwise use greedy baseline
    if best_solution is not None and len(best_solution) <= 256:
        # Verify
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback: return the greedy solution with first_d=8
    r = target - Fraction(1, 8)
    values = [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
