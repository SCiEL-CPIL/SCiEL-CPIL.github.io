import time, random, itertools
import numpy as np

DEADLINE = 78.0  # safety margin under 90s

def _nz(poly):
    return sum(1 for c in poly if c != 0)

def _square_nonzero_count(exponents, coeffs):
    # sparse square: sum over i<=j of 2*[i!=j]*c_i*c_j at e_i+e_j
    n = len(exponents)
    d = {}
    for i in range(n):
        for j in range(i, n):
            s = exponents[i] + exponents[j]
            val = coeffs[i] * coeffs[j]
            if i != j:
                val *= 2
            d[s] = d.get(s, 0) + val
    return sum(1 for v in d.values() if v != 0)

def _validate(exponents, coeffs):
    if len(exponents) != 12:
        return False
    if len(set(exponents)) != 12:
        return False
    if any(c == 0 for c in coeffs):
        return False
    if any(abs(c) > 64 for c in coeffs):
        return False
    if max(exponents) > 40:
        return False
    return True

def _solve_for_cancellation(exponents, coeffs):
    # Try to adjust coefficients to zero out some pair-sums.
    # We fix exponents; find integer coeffs (in [-64,64]) minimizing square sparsity.
    # Strategy: pick a target sum S that has many pairs; force those to cancel.
    n = 12
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    # group pairs by sum
    from collections import defaultdict
    by_sum = defaultdict(list)
    for i, j in pairs:
        by_sum[exponents[i] + exponents[j]].append((i, j))
    # choose the sum with most pairs
    best_sum = max(by_sum, key=lambda s: len(by_sum[s]))
    # We want sum of c_i*c_j over pairs with that sum = 0 (with factor 2 for i!=j)
    # This is a quadratic constraint. Instead, use a linear trick:
    # Set coefficients so that for the chosen sum, the weighted sum cancels.
    # Simpler: use random restarts with local search on coefficients.
    return exponents, coeffs

def _local_search(exponents, coeffs, time_budget=5.0):
    # Hill-climb on coefficients (keep exponents fixed) to reduce square sparsity.
    best_ex = list(exponents)
    best_c = list(coeffs)
    best_score = _square_nonzero_count(best_ex, best_c)
    start = time.time()
    while time.time() - start < time_budget:
        improved = False
        # try changing one coefficient
        for k in range(12):
            for delta in range(-64, 65):
                if delta == 0:
                    continue
                new_c = best_c[k] + delta
                if new_c == 0 or abs(new_c) > 64:
                    continue
                old = best_c[k]
                best_c[k] = new_c
                s = _square_nonzero_count(best_ex, best_c)
                if s < best_score:
                    best_score = s
                    improved = True
                else:
                    best_c[k] = old
        # try swapping two exponents
        for i in range(12):
            for j in range(i+1, 12):
                best_ex[i], best_ex[j] = best_ex[j], best_ex[i]
                s = _square_nonzero_count(best_ex, best_c)
                if s < best_score:
                    best_score = s
                    improved = True
                else:
                    best_ex[i], best_ex[j] = best_ex[j], best_ex[i]
        if not improved:
            break
    return best_ex, best_c, best_score

def _random_exponents(max_deg=40):
    return sorted(random.sample(range(0, max_deg+1), 12))

def _random_coeffs():
    return [random.choice([-1, 1]) * random.randint(1, 64) for _ in range(12)]

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_ex = None
    best_c = None
    best_score = 144  # max possible
    
    # Phase 1: Structured exponent patterns that create many collisions
    # Pattern: arithmetic progressions create many equal pair-sums
    structured_patterns = []
    
    # AP with difference 1: exponents 0..11
    structured_patterns.append(list(range(12)))
    # AP with difference 2
    structured_patterns.append(list(range(0, 24, 2)))
    # Two interleaved APs
    structured_patterns.append(sorted(list(range(0, 12, 2)) + list(range(1, 12, 2))))
    # Symmetric around center
    structured_patterns.append(sorted([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]))
    # Clustered
    structured_patterns.append(sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    # Spread with some collisions
    structured_patterns.append(sorted([0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
    
    for ex in structured_patterns:
        if time.time() - t0 > DEADLINE:
            break
        if not _validate(ex, [1]*12):
            continue
        c = _random_coeffs()
        score = _square_nonzero_count(ex, c)
        if score < best_score:
            best_score = score
            best_ex = list(ex)
            best_c = list(c)
        # Local search
        ex2, c2, s2 = _local_search(ex, c, time_budget=2.0)
        if s2 < best_score:
            best_score = s2
            best_ex = ex2
            best_c = c2
    
    # Phase 2: Random restarts with local search
    while time.time() - t0 < DEADLINE - 2.0:
        ex = _random_exponents(max_deg=40)
        c = _random_coeffs()
        score = _square_nonzero_count(ex, c)
        if score < best_score:
            best_score = score
            best_ex = list(ex)
            best_c = list(c)
        # Local search
        ex2, c2, s2 = _local_search(ex, c, time_budget=1.5)
        if s2 < best_score:
            best_score = s2
            best_ex = ex2
            best_c = c2
    
    # Final validation
    if best_ex is None:
        best_ex = list(range(12))
        best_c = [1]*12
    
    if not _validate(best_ex, best_c):
        best_ex = list(range(12))
        best_c = [1]*12
    
    # Build coefficient list
    result = [0] * (max(best_ex) + 1)
    for e, c in zip(best_ex, best_c):
        result[e] = c
    # Ensure length <= 41 and nonzero final
    if len(result) > 41:
        result = result[:41]
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    if len(result) == 0:
        result = [1]
    
    return result
