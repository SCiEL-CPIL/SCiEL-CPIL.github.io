"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from itertools import product

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 85  # 85s safety margin out of 90s
    best_solution = None
    best_max_den = float('inf')
    
    def greedy_expand(frac, min_d=1, used=None):
        """Greedy expansion with distinctness constraint."""
        terms = []
        r = frac
        min_d = min_d
        used = used or set()
        while r > 0:
            # Find smallest d >= min_d such that 1/d <= r and d not in used
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms
    
    # Get baseline
    baseline = greedy_expand(target)
    if baseline:
        best_solution = baseline[:]
        best_max_den = max(baseline)
    
    def complete_greedy(remaining_frac, used_denoms, min_d_allowed):
        """Complete using greedy, ensuring all denominators are distinct and >= min_d_allowed."""
        terms = []
        r = remaining_frac
        min_d = min_d_allowed
        used = used_denoms.copy()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms
    
    def try_solution(denoms):
        """Try a set of initial denominators and complete greedily."""
        nonlocal best_solution, best_max_den
        if time.time() > deadline:
            return
        remaining = target
        used = set()
        for d in denoms:
            remaining -= Fraction(1, d)
            used.add(d)
            if remaining < 0:
                return
        if remaining == 0:
            if len(denoms) >= 1 and len(denoms) <= 256:
                max_den = max(denoms)
                if max_den < best_max_den:
                    best_max_den = max_den
                    best_solution = denoms[:]
            return
        min_d = max(denoms) + 1 if denoms else 1
        completion = complete_greedy(remaining, used, min_d)
        if completion and len(completion) + len(denoms) <= 256:
            full_solution = denoms + completion
            max_den = max(full_solution)
            if max_den < best_max_den:
                best_max_den = max_den
                best_solution = full_solution[:]
    
    # Strategy 1: Systematic search over first 1-3 denominators
    # First denominator must be at least ceil(31/5) = 7
    for d1 in range(7, 30):
        if time.time() > deadline:
            break
        try_solution([d1])
    
    for d1 in range(7, 15):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, d1 + 20):
            if time.time() > deadline:
                break
            try_solution([d1, d2])
    
    for d1 in range(7, 12):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, d1 + 12):
            if time.time() > deadline:
                break
            for d3 in range(d2 + 1, d2 + 12):
                if time.time() > deadline:
                    break
                try_solution([d1, d2, d3])
    
    # Strategy 2: Randomized search with diverse initial denominators
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            break
        n_init = random.randint(1, 5)
        init_denoms = set()
        for _ in range(n_init):
            # Bias toward smaller denominators
            d = random.randint(7, 100)
            init_denoms.add(d)
        if len(init_denoms) < n_init:
            continue
        try_solution(sorted(init_denoms))
    
    # Strategy 3: Local optimization - try to reduce max denominator
    # For the current best solution, try replacing the largest denominator with alternatives
    if best_solution and best_max_den < float('inf'):
        for _ in range(100):
            if time.time() > deadline:
                break
            current = best_solution[:]
            max_idx = current.index(max(current))
            max_den = current[max_idx]
            
            # Try replacing the max denominator with a smaller one
            # and see if we can complete the rest
            for new_d in range(7, max_den):
                if time.time() > deadline:
                    break
                if new_d in current:
                    continue
                # Remove the max denominator and add new_d
                remaining = target
                used = set()
                new_solution = []
                for d in current:
                    if d != max_den:
                        remaining -= Fraction(1, d)
                        used.add(d)
                        new_solution.append(d)
                
                # Now we need to represent remaining + 1/max_den - 1/new_d
                target_remaining = remaining + Fraction(1, max_den) - Fraction(1, new_d)
                if target_remaining <= 0:
                    continue
                
                # Try to complete with greedy
                min_d = max(new_solution) + 1 if new_solution else 1
                # Also ensure new_d is not used
                if new_d in used:
                    continue
                used.add(new_d)
                new_solution.append(new_d)
                
                completion = complete_greedy(target_remaining, used, min_d)
                if completion and len(new_solution) + len(completion) <= 256:
                    full_solution = new_solution + completion
                    max_den_new = max(full_solution)
                    if max_den_new < best_max_den:
                        best_max_den = max_den_new
                        best_solution = full_solution[:]
                        break
    
    # Strategy 4: More aggressive search with different heuristics
    # Try to find solutions where denominators are more evenly distributed
    for d1 in range(7, 25):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, min(d1 + 30, 100)):
            if time.time() > deadline:
                break
            remaining = target - Fraction(1, d1) - Fraction(1, d2)
            if remaining <= 0:
                continue
            # Try different completions
            for d3_start in range(d2 + 1, min(d2 + 15, 80)):
                if time.time() > deadline:
                    break
                try_solution([d1, d2, d3_start])
    
    if best_solution is None:
        # Fallback
        r, values = target, []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
# EVOLVE-BLOCK-END
