import math
import time
import sys

def construct():
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute GCDs
    gcd_cache = {}
    for i in range(n_moduli):
        m1 = moduli[i]
        for j in range(i, n_moduli):
            m2 = moduli[j]
            g = math.gcd(m1, m2)
            gcd_cache[(m1, m2)] = g
            if m1 != m2:
                gcd_cache[(m2, m1)] = g

    # Precompute weights
    weights = [1.0 / m for m in moduli]
    
    # Precompute suffix sums of weights for a fast, loose upper bound
    suffix_sums = [0.0] * (n_moduli + 1)
    for i in range(n_moduli - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]

    def is_valid_r(m, r, current_pairs):
        for m2, r2 in current_pairs:
            g = gcd_cache.get((m, m2), 1)
            if (r - r2) % g == 0:
                return False
        return True

    def calc_tight_bound(current_pairs, start_idx):
        ub = 0.0
        for i in range(start_idx, n_moduli):
            m = moduli[i]
            found = False
            for r in range(m):
                if is_valid_r(m, r, current_pairs):
                    found = True
                    break
            if found:
                ub += weights[i]
        return ub

    def greedy_init():
        selected = []
        for m in moduli:
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in selected:
                    g = gcd_cache.get((m, m2), 1)
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append((m, r))
                    found = True
                    break
        return selected

    initial_sol = greedy_init()
    best_score = sum(1.0 / m for m, _ in initial_sol)
    best_sol = list(initial_sol)
    
    start_time = time.time()
    time_limit = 118.0

    sys.setrecursionlimit(500)

    def solve(idx, current_score, current_pairs):
        nonlocal best_score, best_sol
        
        if time.time() - start_time > time_limit:
            return

        if idx == n_moduli:
            if current_score > best_score:
                best_score = current_score
                best_sol = list(current_pairs)
            return

        loose_ub = current_score + suffix_sums[idx]
        if loose_ub <= best_score:
            return

        if idx < 50 or (loose_ub - best_score < 0.05):
            tight_ub = current_score + calc_tight_bound(current_pairs, idx)
            if tight_ub <= best_score:
                return

        m = moduli[idx]
        w = weights[idx]

        # Try Include first (higher weight potential)
        for r in range(m):
            if is_valid_r(m, r, current_pairs):
                new_pairs = current_pairs + [(m, r)]
                solve(idx + 1, current_score + w, new_pairs)

        # Try Skip
        if current_score + suffix_sums[idx + 1] > best_score:
            solve(idx + 1, current_score, current_pairs)

    solve(0, 0.0, [])

    return [[m, r] for m, r in best_sol]
