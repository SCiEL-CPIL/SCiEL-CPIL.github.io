"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 80  # 80s safety margin out of 90s
    best_solution = None
    best_max_den = float('inf')
    
    def complete_greedy(remaining_frac, used_denoms, min_d_allowed):
        """Complete using greedy, ensuring all denominators are distinct and >= min_d_allowed."""
        terms = []
        r = remaining_frac
        min_d = min_d_allowed
        used = used_denoms
        while r > 0:
            # d >= ceil(1/r)
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
    
    def complete_balanced(remaining_frac, used_denoms, min_d_allowed, max_den_limit=None):
        """Try to complete with denominators spread out to minimize the max.
        At each step, instead of taking the smallest d, try a range of d values
        and pick the one that leads to a smaller eventual max."""
        if remaining_frac <= 0:
            return []
        r = remaining_frac
        min_d = min_d_allowed
        used = used_denoms.copy()
        terms = []
        
        # Estimate how many terms we'll need
        # If we use n terms each ~1/d_avg, then n/d_avg ≈ r
        # So d_avg ≈ n/r. We want to spread them.
        
        # Try a few strategies: greedy, and slightly-larger-first
        best_terms = None
        best_max = float('inf')
        
        # Strategy A: pure greedy
        terms_a = []
        r_a = remaining_frac
        min_da = min_d
        used_a = used_denoms.copy()
        while r_a > 0:
            d = (r_a.denominator + r_a.numerator - 1) // r_a.numerator
            if d < min_da:
                d = min_da
            while d in used_a:
                d += 1
            terms_a.append(d)
            r_a -= Fraction(1, d)
            min_da = d + 1
            used_a.add(d)
            if max_den_limit and d > max_den_limit:
                terms_a = None
                break
        if terms_a:
            best_terms = terms_a
            best_max = max(terms_a)
        
        # Strategy B: at each step, try d, d+1, d+2, d+3 and pick the one
        # that minimizes the eventual max (lookahead 1 level)
        r_b = remaining_frac
        min_db = min_d
        used_b = used_denoms.copy()
        terms_b = []
        success = True
        while r_b > 0:
            d_min = (r_b.denominator + r_b.numerator - 1) // r_b.numerator
            if d_min < min_db:
                d_min = min_db
            # Try a few candidates
            candidates = []
            for dd in range(d_min, d_min + 6):
                if dd in used_b:
                    continue
                if max_den_limit and dd > max_den_limit:
                    continue
                # Check 1/dd <= r_b
                if Fraction(1, dd) > r_b:
                    continue
                candidates.append(dd)
            if not candidates:
                success = False
                break
            # Pick candidate that leaves a "nicer" remainder
            # Heuristic: pick the one where the next greedy step gives smallest d
            best_cand = candidates[0]
            best_next_d = float('inf')
            for cand in candidates:
                r_next = r_b - Fraction(1, cand)
                if r_next <= 0:
                    # This finishes it
                    if cand < best_next_d:
                        best_next_d = cand
                        best_cand = cand
                    continue
                d_next = (r_next.denominator + r_next.numerator - 1) // r_next.numerator
                if d_next < cand + 1:
                    d_next = cand + 1
                if d_next < best_next_d:
                    best_next_d = d_next
                    best_cand = cand
            terms_b.append(best_cand)
            r_b -= Fraction(1, best_cand)
            min_db = best_cand + 1
            used_b.add(best_cand)
        if success and r_b == 0:
            max_b = max(terms_b)
            if max_b < best_max:
                best_terms = terms_b
                best_max = max_b
        
        return best_terms
    
    def try_solution(denoms):
        """Try a set of initial denominators and complete."""
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
            if 1 <= len(denoms) <= 256:
                max_den = max(denoms)
                if max_den < best_max_den:
                    best_max_den = max_den
                    best_solution = denoms[:]
            return
        min_d = max(denoms) + 1 if denoms else 1
        # Try both greedy and balanced completion
        for completion in [complete_greedy(remaining, used, min_d),
                          complete_balanced(remaining, used, min_d, best_max_den)]:
            if completion and len(completion) + len(denoms) <= 256:
                full_solution = denoms + completion
                max_den = max(full_solution)
                if max_den < best_max_den:
                    best_max_den = max_den
                    best_solution = full_solution[:]
    
    # Strategy 1: Systematic search over first 1-3 denominators
    for d1 in range(7, 35):
        if time.time() > deadline:
            break
        try_solution([d1])
    
    for d1 in range(7, 16):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, d1 + 25):
            if time.time() > deadline:
                break
            try_solution([d1, d2])
    
    for d1 in range(7, 13):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, d1 + 15):
            if time.time() > deadline:
                break
            for d3 in range(d2 + 1, d2 + 15):
                if time.time() > deadline:
                    break
                try_solution([d1, d2, d3])
    
    # Strategy 2: Randomized search
    random.seed(42)
    for _ in range(3000):
        if time.time() > deadline:
            break
        n_init = random.randint(1, 5)
        init_denoms = set()
        for _ in range(n_init):
            d = random.randint(7, 120)
            init_denoms.add(d)
        if len(init_denoms) < n_init:
            continue
        try_solution(sorted(init_denoms))
    
    # Strategy 3: Local optimization - try replacing any denominator
    if best_solution and best_max_den < float('inf'):
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            current = best_solution[:]
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                old_d = current[i]
                # Try replacing with smaller values
                for new_d in range(7, old_d):
                    if new_d in current:
                        continue
                    # Remove old_d, add new_d, re-complete
                    remaining = target
                    used = set()
                    new_solution = []
                    for j, d in enumerate(current):
                        if j != i:
                            remaining -= Fraction(1, d)
                            used.add(d)
                            new_solution.append(d)
                    # Now need to represent: remaining + 1/old_d - 1/new_d
                    target_rem = remaining + Fraction(1, old_d) - Fraction(1, new_d)
                    if target_rem <= 0:
                        continue
                    if new_d in used:
                        continue
                    used.add(new_d)
                    new_solution.append(new_d)
                    min_d = max(new_solution) + 1 if new_solution else 1
                    completion = complete_greedy(target_rem, used, min_d)
                    if completion and len(new_solution) + len(completion) <= 256:
                        full_solution = new_solution + completion
                        max_den_new = max(full_solution)
                        if max_den_new < best_max_den:
                            best_max_den = max_den_new
                            best_solution = full_solution[:]
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try adding a new small denominator and removing a large one
                # This is more complex; skip for now
                break
    
    # Strategy 4: Binary search on max denominator
    # Check if a solution exists with all denominators <= M
    def check_max_den(M):
        """Try to find a solution with all denominators <= M."""
        # Use the current best solution's structure as a guide
        # Try greedy with cap
        r = target
        min_d = 1
        used = set()
        terms = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            if d > M:
                return None
            while d in used:
                d += 1
                if d > M:
                    return None
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms
    
    # Try to find the minimum M
    lo, hi = 7, best_max_den
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = check_max_den(mid)
        if sol and 1 <= len(sol) <= 256:
            hi = mid
        else:
            lo = mid + 1
    # Try the found boundary
    for M in range(lo, min(lo + 5, hi + 1)):
        if time.time() > deadline:
            break
        sol = check_max_den(M)
        if sol and 1 <= len(sol) <= 256:
            max_den = max(sol)
            if max_den < best_max_den:
                best_max_den = max_den
                best_solution = sol[:]
    
    # Strategy 5: More systematic 4-denominator search for small ranges
    for d1 in range(7, 11):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, d1 + 10):
            if time.time() > deadline:
                break
            for d3 in range(d2 + 1, d2 + 10):
                if time.time() > deadline:
                    break
                for d4 in range(d3 + 1, d3 + 10):
                    if time.time() > deadline:
                        break
                    try_solution([d1, d2, d3, d4])
    
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
