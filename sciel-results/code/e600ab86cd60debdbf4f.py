"""Optimized program for Egyptian fraction of 5/31; minimizes largest denominator."""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # First, get a quick baseline with greedy
    def greedy(r, min_den, max_terms=256):
        values = []
        used = set()
        for _ in range(max_terms):
            if r == 0:
                return values
            d = (r.denominator + r.numerator - 1) // r.numerator
            d = max(d, min_den)
            if d in used:
                # Try next
                d += 1
                while d in used:
                    d += 1
            if 1/d > r:
                return None
            values.append(d)
            used.add(d)
            r -= Fraction(1, d)
            min_den = d + 1
        return values if r == 0 else None
    
    # Quick greedy baseline
    baseline = greedy(target, 7)
    best_max = max(baseline) if baseline else 10000000
    best_solution = baseline
    
    # Now try to find better solutions with bounded max denominator
    # Use DFS with pruning
    def dfs_bounded(r, min_den, max_den, current, depth, max_depth=256):
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
        
        # Lower bound on next denominator
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Upper bound: we need at least 1/d <= r, so d >= ceil(1/r)
        # Also, remaining terms must sum to r, and each term <= 1/min_den
        # If we have (max_depth - depth) terms left, max possible sum is sum of 1/d for d from min_den to min_den + (max_depth-depth-1)
        # This is a loose bound, skip for now
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            # Pruning: if new_r > 0, check if it's possible to complete
            # with remaining denominators in [d+1, max_den]
            if new_r > 0:
                # Max possible sum with remaining terms: sum of 1/(d+1) + 1/(d+2) + ... 
                # But we have at most (max_depth - depth - 1) terms
                # Simple check: 1/(d+1) should be >= new_r / (max_depth - depth - 1) if we need multiple terms
                # Actually, just check if new_r <= sum of 1/k for k=d+1 to max_den (bounded by remaining depth)
                remaining_terms = max_depth - depth - 1
                if remaining_terms <= 0:
                    continue
                # Upper bound on achievable sum with remaining_terms terms from d+1 to max_den
                # The max sum is 1/(d+1) + 1/(d+2) + ... + 1/(d+remaining_terms)
                # But if d+remaining_terms > max_den, cap it
                max_k = min(d + remaining_terms, max_den)
                if max_k <= d:
                    continue
                # Quick upper bound: remaining_terms / (d+1)
                if new_r > Fraction(remaining_terms, d + 1):
                    continue
                # Lower bound: 1/max_den * remaining_terms (very loose, skip)
            
            current.append(d)
            if dfs_bounded(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try decreasing max_den values to find the minimum
    # Start from current best and go down
    search_start = min(best_max - 1, 300)
    search_end = 15  # theoretical minimum is around 15-20
    
    for max_den in range(search_start, search_end - 1, -1):
        if time.time() > deadline:
            break
        # Try to find a solution with all denominators <= max_den
        if dfs_bounded(target, 7, max_den, [], 0, 256):
            if best_max <= max_den:
                break
        # If we found a solution, best_max was updated, so we can break
        if best_max <= max_den:
            break
    
    # Also try a few specific strategies
    # Try pairs and triples of small denominators
    for d1 in range(7, min(20, best_max)):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, min(30, best_max)):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            # Try to complete with bounded denominators
            for max_d in range(min(best_max - 1, 50), d2, -1):
                if time.time() > deadline:
                    break
                if dfs_bounded(r2, d2 + 1, max_d, [d1, d2], 2, 256):
                    if best_max <= max_d:
                        break
    
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback
    return baseline
