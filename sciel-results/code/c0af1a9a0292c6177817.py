import numpy as np
import time
import random
import sys
from itertools import combinations, product
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def square_count(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    best_score = float('inf')
    best_poly = None

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        sc = square_count(coeffs)
        if sc < best_score:
            best_score = sc
            best_poly = list(coeffs)

    # Precompute: for a given support (sorted list of 12 positions),
    # build the pair-sum structure.
    # Square coefficient at position k = sum over pairs (i,j) with pos_i + pos_j = k of c_i * c_j.
    # Diagonal pairs (i,i) contribute c_i^2 (always positive if we fix sign, but sign matters for cross terms).
    # Actually c_i^2 is always positive regardless of sign. Cross terms c_i*c_j for i!=j can be positive or negative.
    
    # For a fixed support, we want to assign coefficients c_i (nonzero, in [-64,64]) 
    # to minimize the number of k where the sum is nonzero.
    
    # Key structural insight: 
    # - Diagonal terms (i,i) are at position 2*pos_i. These are ALWAYS nonzero (c_i^2 > 0).
    # - So minimum possible nonzero count >= number of distinct 2*pos_i values.
    # - Cross terms at position pos_i + pos_j (i<j) can potentially cancel with other cross terms at same position.
    
    # Let's build for a support:
    # 1. Diagonal positions: diag_pos = {2*p for p in support}
    # 2. Cross positions: for i<j, cross_pos = pos_i + pos_j
    # 3. For each position k, the set of contributing pairs.
    
    # For cancellation to work at a cross position k, we need at least 2 cross pairs (or 1 cross + 1 diag if 2*pos_i = pos_j+pos_k).
    # But diag contributions are positive and fixed magnitude (c_i^2), cross contributions are c_i*c_j.
    # Cancellation requires: sum of cross contributions + (diag contribution if any) = 0.
    
    # Simplified approach for branch-and-bound:
    # Focus on supports where many cross-term positions have multiplicity >= 2,
    # enabling sign-based cancellation.
    
    def analyze_support(support):
        """Returns (min_possible_nonzero, cross_groups, diag_positions)."""
        diag_positions = set(2 * p for p in support)
        
        # Build cross-term groups: position -> list of (i, j) pairs
        cross_groups = defaultdict(list)
        for i in range(12):
            for j in range(i + 1, 12):
                k = support[i] + support[j]
                cross_groups[k].append((i, j))
        
        # A cross position k is "cancelable" if:
        # - len(cross_groups[k]) >= 2 (multiple cross pairs can cancel each other), OR
        # - k is in diag_positions and len(cross_groups[k]) >= 1 (cross can cancel diag)
        
        # Minimum nonzeros: all diag positions are nonzero.
        # Cross positions with only 1 pair and not at diag position: always nonzero.
        # Cross positions with >= 2 pairs or at diag position: potentially zero.
        
        always_nonzero = len(diag_positions)
        for k, pairs in cross_groups.items():
            if len(pairs) == 1 and k not in diag_positions:
                always_nonzero += 1
        
        return always_nonzero, cross_groups, diag_positions

    def solve_support_bnb(support, time_limit=0.5):
        """Branch and bound over sign/magnitude assignments for a given support."""
        t0 = time.time()
        min_nonzero, cross_groups, diag_positions = analyze_support(support)
        
        if min_nonzero >= best_score:
            return None
        
        # For BnB, we only need to consider signs (±1) and small magnitudes.
        # Diagonal terms: c_i^2. If we use magnitude m_i, diag contribution is m_i^2.
        # Cross term at (i,j): c_i * c_j = s_i * s_j * m_i * m_j.
        
        # Simplification: first try all-ones magnitudes with sign search.
        # If that doesn't beat best, try a few magnitude variations.
        
        # Sign search via BnB:
        # Variables: s_i in {+1, -1} for i in 0..11
        # Objective: minimize count of k where sum_{(i,j) in cross_groups[k]} s_i*s_j + (diag if k in diag) != 0
        
        # Upper bound: current best_score
        # Lower bound: min_nonzero
        
        # Greedy initialization: try all +1, then flip one at a time.
        coeffs = [1] * 12
        max_deg = max(support)
        coeff_arr = np.zeros(max_deg + 1, dtype=np.int64)
        for i, p in enumerate(support):
            coeff_arr[p] = coeffs[i]
        
        def eval_signs(signs):
            nonlocal coeff_arr
            for i in range(12):
                coeff_arr[support[i]] = signs[i]
            sq = np.convolve(coeff_arr, coeff_arr)
            return int(np.count_nonzero(sq))
        
        # Greedy: start with all +1, flip to improve
        signs = [1] * 12
        current = eval_signs(signs)
        
        # Local search on signs
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            for i in range(12):
                signs[i] = -signs[i]
                new = eval_signs(signs)
                if new < current:
                    current = new
                    improved = True
                else:
                    signs[i] = -signs[i]  # revert
        
        if current < best_score:
            # Build coeffs
            coeffs = [0] * (max_deg + 1)
            for i, p in enumerate(support):
                coeffs[p] = signs[i]
            # Trim trailing zeros
            while coeffs and coeffs[-1] == 0:
                coeffs.pop()
            if len(coeffs) >= 1 and coeffs[-1] != 0:
                return coeffs
        
        # Try some magnitude variations
        if time.time() - t0 < time_limit:
            for mag_pattern in [
                [1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,2,1,1,1,1,1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
                [1,1,2,1,1,2,1,1,2,1,1,2],
            ]:
                if time.time() - t0 > time_limit:
                    break
                # For each magnitude pattern, do sign search
                signs = [1] * 12
                current = float('inf')
                # Evaluate with magnitudes
                def eval_signs_mag(s):
                    nonlocal coeff_arr
                    for i in range(12):
                        coeff_arr[support[i]] = s[i] * mag_pattern[i]
                    sq = np.convolve(coeff_arr, coeff_arr)
                    return int(np.count_nonzero(sq))
                
                current = eval_signs_mag(signs)
                improved = True
                while improved and time.time() - t0 < time_limit:
                    improved = False
                    for i in range(12):
                        signs[i] = -signs[i]
                        new = eval_signs_mag(signs)
                        if new < current:
                            current = new
                            improved = True
                        else:
                            signs[i] = -signs[i]
                
                if current < best_score:
                    coeffs = [0] * (max_deg + 1)
                    for i, p in enumerate(support):
                        coeffs[p] = signs[i] * mag_pattern[i]
                    while coeffs and coeffs[-1] == 0:
                        coeffs.pop()
                    if len(coeffs) >= 1 and coeffs[-1] != 0:
                        return coeffs
        
        return None

    # Generate candidate supports
    # Strategy: arithmetic progressions, geometric progressions, and random with collision bias
    
    candidate_supports = []
    
    # Arithmetic progressions: {a, a+d, a+2d, ..., a+11d}
    for a in range(0, 4):
        for d in range(1, 4):
            support = [a + i * d for i in range(12)]
            if max(support) <= 40:
                candidate_supports.append(tuple(support))
    
    # Two interleaved arithmetic progressions
    for a in range(0, 3):
        for d1 in range(1, 3):
            for d2 in range(d1 + 1, d1 + 3):
                support = sorted([a + i * d1 for i in range(6)] + [a + 1 + i * d2 for i in range(6)])
                if len(set(support)) == 12 and max(support) <= 40:
                    candidate_supports.append(tuple(support))
    
    # Geometric: {0, 1, 2, 4, 8, 16, ...} plus fillers
    for base in [2, 3]:
        support = []
        p = 0
        for i in range(12):
            if p <= 40:
                support.append(p)
            else:
                support.append(40 - (11 - i))
            p = p * base if p < 40 else 40
        support = sorted(set(support))
        while len(support) < 12:
            support.append(random.randint(0, 40))
        support = sorted(set(support))[:12]
        if len(support) == 12:
            candidate_supports.append(tuple(support))
    
    # Random supports with collision bias
    random.seed(42)
    for _ in range(200):
        # Cluster around a few centers
        centers = random.sample(range(0, 41), random.randint(2, 4))
        support = []
        for c in centers:
            for offset in random.sample(range(-3, 4), random.randint(2, 5)):
                pos = c + offset
                if 0 <= pos <= 40 and pos not in support:
                    support.append(pos)
        while len(support) < 12:
            pos = random.randint(0, 40)
            if pos not in support:
                support.append(pos)
        support = sorted(support[:12])
        if len(set(support)) == 12:
            candidate_supports.append(tuple(support))
    
    # Deduplicate
    candidate_supports = list(set(candidate_supports))
    
    # Sort by min_possible_nonzero (ascending) to find good ones first
    def support_key(s):
        min_nz, _, _ = analyze_support(s)
        return min_nz
    
    candidate_supports.sort(key=support_key)
    
    # Process supports
    processed = 0
    for support in candidate_supports:
        if time.time() > deadline:
            break
        
        min_nz, _, _ = analyze_support(support)
        if min_nz >= best_score:
            continue
        
        result = solve_support_bnb(support, time_limit=0.3)
        if result is not None:
            try_poly(result)
        
        processed += 1
        if processed % 50 == 0:
            sys.stdout.write(f"Processed {processed} supports, best={best_score}\n")
            sys.stdout.flush()
    
    # If no good result, try a broader random search
    if best_score == float('inf') or best_score > 15:
        random.seed(123)
        for _ in range(500):
            if time.time() > deadline:
                break
            support = tuple(sorted(random.sample(range(41), 12)))
            min_nz, _, _ = analyze_support(support)
            if min_nz < best_score:
                result = solve_support_bnb(support, time_limit=0.2)
                if result is not None:
                    try_poly(result)
    
    # Final fallback: simple structured polynomial
    if best_poly is None:
        # P(x) = 1 + x + x^2 + ... + x^11 (12 consecutive terms)
        best_poly = [1] * 12
        if not is_valid(best_poly):
            best_poly = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return best_poly
