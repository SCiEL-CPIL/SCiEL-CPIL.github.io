"""Improved program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def try_greedy(r, min_den, max_den, max_terms=255):
        """Complete Egyptian fraction greedily. Return list or None."""
        values = []
        for _ in range(max_terms):
            if r == 0:
                return values
            if time.time() > deadline:
                return None
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None
    
    # Phase 1: Quick greedy search to establish a good baseline
    for first_d in range(7, 60):
        if time.time() > deadline:
            break
        r = target - Fraction(1, first_d)
        if r < 0:
            continue
        for max_den in [200, 400, 800, 1600, 3200]:
            sol = try_greedy(r, first_d + 1, max_den)
            if sol is not None:
                full_sol = [first_d] + sol
                if len(full_sol) <= 256 and len(set(full_sol)) == len(full_sol):
                    m = max(full_sol)
                    if m < best_max:
                        best_max = m
                        best_solution = full_sol
    
    # Phase 2: Bounded DFS to find better solutions
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
            return True
        if depth >= max_depth:
            return False
        if r <= 0:
            return False
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Pruning: if even with all remaining terms at min_den we can't reach r
        # Max possible sum from remaining terms: (max_depth - depth) / min_den
        # But we need exactly r, and terms must be distinct and >= min_den
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try DFS with decreasing max_den bounds
    search_bounds = [best_max - 1, 500, 400, 300, 250, 200]
    for search_max in search_bounds:
        if time.time() > deadline:
            break
        if search_max < 10:
            break
        # Try different starting denominators
        for first_d in range(7, min(30, search_max)):
            if time.time() > deadline:
                break
            r = target - Fraction(1, first_d)
            if r < 0:
                continue
            if dfs(r, first_d + 1, search_max, [first_d], 1):
                if best_max <= search_max:
                    break
    
    # Final verification and return
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution) and all(1 <= d <= 10000000 for d in best_solution):
            return best_solution
    
    # Fallback
    r = target - Fraction(1, 8)
    values = [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
