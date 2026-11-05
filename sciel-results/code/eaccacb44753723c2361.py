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
    
    def can_complete_greedy(r, min_den, max_den, max_terms=256):
        """Try to complete r using greedy with bounds. Return list or None."""
        if r == 0:
            return []
        values = []
        for _ in range(max_terms):
            if time.time() > deadline:
                return None
            if r == 0:
                return values
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
    
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        """Bounded DFS with strong pruning."""
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
        if depth >= max_depth or r <= 0:
            return False
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Pruning: if we use the smallest possible d, the remaining value must be
        # representable with remaining terms.
        remaining_terms = max_depth - depth - 1
        if remaining_terms <= 0:
            return False
        # Max possible sum with remaining terms is roughly remaining_terms / (d_start + 1)
        if r > Fraction(remaining_terms, d_start + 1):
            return False
        
        # Additional pruning: lower bound on sum with remaining terms
        # The smallest remaining terms are d_start+1, d_start+2, ...
        # A rough lower bound: remaining_terms / (d_start + remaining_terms)
        if r < Fraction(1, d_start + 1) * remaining_terms / (remaining_terms + 1):
            # This is a weak lower bound, skip for now
            pass
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break  # Since we're increasing d, new_r will only get more negative
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Phase 1: Quick greedy to get an initial upper bound
    for d1 in range(7, 50):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        sol = can_complete_greedy(r1, d1 + 1, 5000)
        if sol is not None and len(sol) + 1 <= 256:
            full_sol = [d1] + sol
            m = max(full_sol)
            if m < best_max:
                best_max = m
                best_solution = full_sol
    
    # Phase 2: Systematic search over first two denominators with greedy completion
    for d1 in range(7, 40):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        d2_min = (r1.denominator + r1.numerator - 1) // r1.numerator
        d2_min = max(d2_min, d1 + 1)
        d2_max = min(d2_min + 50, best_max - 1 if best_max < float('inf') else 2000)
        
        for d2 in range(d2_min, d2_max + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                break
            
            max_den = best_max - 1 if best_max < float('inf') else 2000
            if max_den < d2 + 1:
                continue
            sol = can_complete_greedy(r2, d2 + 1, max_den)
            if sol is not None and len(sol) + 2 <= 256:
                full_sol = [d1, d2] + sol
                m = max(full_sol)
                if m < best_max:
                    best_max = m
                    best_solution = full_sol
    
    # Phase 3: DFS with tight bounds to find better solutions
    if best_max < float('inf'):
        # Try decreasing max_den values
        for search_max in range(min(best_max - 1, 500), 30, -1):
            if time.time() > deadline:
                break
            if dfs(target, 1, search_max, [], 0):
                break
    
    # Phase 4: Further refinement with smaller max_den
    if best_max < float('inf') and best_max > 50:
        for search_max in range(min(best_max - 1, 200), 40, -1):
            if time.time() > deadline:
                break
            if dfs(target, 1, search_max, [], 0):
                break
    
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback: simple greedy
    r = target
    values = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
