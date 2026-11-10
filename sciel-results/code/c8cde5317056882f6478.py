import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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

    def count_nonzero_square(coeffs):
        # Fast convolution using list-based approach
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeffs[j]
                if cj != 0:
                    sq[i + j] += ci * cj
        return sum(1 for x in sq if x != 0)

    best_score = float('inf')
    best_poly = None

    # Baseline: try a few simple structured polynomials first
    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = count_nonzero_square(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Quick baseline checks
    try_poly([1] * 12)
    try_poly([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    try_poly([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

    # Constraint-based backtracking search
    # We fix a support (12 positions) and try to find coefficients that minimize
    # the number of nonzero terms in the square.
    
    # Key insight: For a polynomial with support S = {s_1 < s_2 < ... < s_12},
    # the square has degrees from 2*s_1 to 2*s_12.
    # The coefficient of x^k in the square is sum_{i+j=k, i,j in S} a_i * a_j.
    # We want as many of these sums to be zero as possible.
    
    # For small supports (low degree range), we can do exhaustive coefficient search.
    # For larger supports, we use backtracking with pruning.

    def backtrack_search(support, time_limit):
        """
        Given a fixed support (list of 12 positions), find coefficients
        that minimize nonzero terms in the square.
        Uses backtracking with constraint propagation.
        """
        n_terms = 12
        # Degree range of the square
        min_deg = 2 * support[0]
        max_deg = 2 * support[-1]
        total_degs = max_deg - min_deg + 1
        
        # Precompute: for each degree k in [min_deg, max_deg], which pairs (i,j) in support satisfy i+j=k
        # Map degree offset to list of (idx_i, idx_j) where support[idx_i] + support[idx_j] = k
        deg_to_pairs = {}
        for i in range(n_terms):
            for j in range(i, n_terms):  # i <= j to avoid double counting
                deg = support[i] + support[j]
                if deg not in deg_to_pairs:
                    deg_to_pairs[deg] = []
                deg_to_pairs[deg].append((i, j))
        
        # Degrees that are "forced nonzero" because they have only one pair (i,i) or (i,j) with i!=j
        # Actually, a degree is forced nonzero if the only way to form it is a single term squared
        # and that coefficient is nonzero. But we're choosing coefficients, so we need to think differently.
        
        # A degree k is "trivially zero" if no pair sums to k.
        # A degree k is "potentially zero" if there are multiple pairs that can cancel.
        
        # For backtracking: assign coefficients one by one.
        # At each step, check which degrees are now "determined" and must be zero.
        
        # Simpler approach for small degree ranges: enumerate coefficient values.
        # If total_degs <= 25 and n_terms is small, try values from a small set.
        
        # Strategy: Try coefficient values from a small candidate set.
        # For each position, try values from {-2, -1, 0, 1, 2} or similar.
        # But we need exactly 12 nonzero terms, so all coefficients in support must be nonzero.
        
        # Let's try a smarter approach: 
        # 1. First, determine which degrees in the square MUST be nonzero.
        #    - The min degree (2*support[0]) = a_0^2, always nonzero if a_0 != 0.
        #    - The max degree (2*support[-1]) = a_11^2, always nonzero if a_11 != 0.
        #    - Any degree that can only be formed by a single pair (i,j) where i==j must be a_i^2 != 0.
        #    - Any degree that can only be formed by a single pair (i,j) where i!=j must be 2*a_i*a_j != 0.
        
        # Compute "forced nonzero" degrees
        forced_nonzero_degs = set()
        for k in range(min_deg, max_deg + 1):
            if k not in deg_to_pairs:
                continue
            pairs = deg_to_pairs[k]
            if len(pairs) == 1:
                i, j = pairs[0]
                # This degree is determined by a single pair
                forced_nonzero_degs.add(k)
        
        # The minimum possible nonzero count is at least |forced_nonzero_degs|
        min_possible = len(forced_nonzero_degs)
        
        if min_possible >= best_score:
            return  # Can't beat current best
        
        # Now try to find coefficients that make all other degrees zero.
        # This is a system of quadratic equations.
        
        # For small problems, try random coefficient assignments with the constraint
        # that all "free" degrees (not forced nonzero) are zero.
        
        free_degs = [k for k in range(min_deg, max_deg + 1) 
                     if k in deg_to_pairs and k not in forced_nonzero_degs]
        
        if not free_degs:
            # All degrees are forced nonzero, so the score is min_possible
            # Just need any valid assignment
            coeffs = [0] * (support[-1] + 1)
            for i, pos in enumerate(support):
                coeffs[pos] = 1
            try_poly(coeffs)
            return
        
        # Try to solve: for each free degree k, sum of a_i * a_j over pairs = 0
        # This is hard in general. Let's use a heuristic: try small integer coefficients.
        
        # Candidate coefficient values: prioritize small values for more cancellation opportunities
        candidate_vals = [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
        
        # Randomized search over coefficient assignments
        best_local_score = min_possible
        best_local_coeffs = None
        
        num_attempts = 2000
        attempt = 0
        while attempt < num_attempts and (time.time() - start_time) < time_limit:
            attempt += 1
            
            # Randomly assign coefficients from candidate values
            coeff_vals = [random.choice(candidate_vals) for _ in range(n_terms)]
            
            # Build the full coefficient array
            coeffs = [0] * (support[-1] + 1)
            for i, pos in enumerate(support):
                coeffs[pos] = coeff_vals[i]
            
            # Compute square and count nonzeros
            n = len(coeffs)
            sq = [0] * (2 * n - 1)
            for i in range(n):
                ci = coeffs[i]
                if ci == 0:
                    continue
                for j in range(n):
                    cj = coeffs[j]
                    if cj != 0:
                        sq[i + j] += ci * cj
            
            score = sum(1 for x in sq if x != 0)
            
            if score < best_local_score:
                best_local_score = score
                best_local_coeffs = list(coeffs)
                if score < best_score:
                    best_score = score
                    best_poly = list(coeffs)
            
            if best_local_score == min_possible:
                break  # Can't do better
        
        # Local refinement: try to fix individual coefficients
        if best_local_coeffs is not None and best_local_score > min_possible:
            for _ in range(500):
                if (time.time() - start_time) > time_limit:
                    break
                # Pick a random position and try different values
                idx = random.randint(0, n_terms - 1)
                old_val = best_local_coeffs[support[idx]]
                for new_val in candidate_vals:
                    if new_val == old_val:
                        continue
                    best_local_coeffs[support[idx]] = new_val
                    n = len(best_local_coeffs)
                    sq = [0] * (2 * n - 1)
                    for i in range(n):
                        ci = best_local_coeffs[i]
                        if ci == 0:
                            continue
                        for j in range(n):
                            cj = best_local_coeffs[j]
                            if cj != 0:
                                sq[i + j] += ci * cj
                    score = sum(1 for x in sq if x != 0)
                    if score < best_local_score:
                        best_local_score = score
                        if score < best_score:
                            best_score = score
                            best_poly = list(best_local_coeffs)
                    else:
                        best_local_coeffs[support[idx]] = old_val
                        break  # Keep old value if new one doesn't help

    # Generate candidate supports to search
    # Focus on supports with small degree ranges to make the constraint system tractable
    
    # Strategy 1: Consecutive or near-consecutive supports
    supports_to_try = []
    
    # Consecutive blocks
    for start in range(0, 30):
        support = list(range(start, start + 12))
        supports_to_try.append(support)
    
    # Alternating patterns
    for start in range(0, 20):
        support = [start + 2*i for i in range(12)]
        supports_to_try.append(support)
    
    # Symmetric patterns
    for center in range(6, 20):
        support = sorted(set([center - 5, center - 4, center - 3, center - 2, center - 1, center,
                               center + 1, center + 2, center + 3, center + 4, center + 5, center + 6]))
        if len(support) == 12 and support[-1] < 41:
            supports_to_try.append(support)
    
    # Random supports with small degree range
    random.seed(42)
    for _ in range(200):
        max_span = random.randint(12, 25)
        start = random.randint(0, 41 - max_span - 1)
        # Choose 12 positions in [start, start + max_span]
        available = list(range(start, start + max_span + 1))
        if len(available) >= 12:
            support = sorted(random.sample(available, 12))
            supports_to_try.append(support)
    
    # Deduplicate
    seen = set()
    unique_supports = []
    for s in supports_to_try:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
    
    # Sort by degree range (smaller first)
    unique_supports.sort(key=lambda s: s[-1] - s[0])
    
    # Search each support with time budget
    for support in unique_supports:
        if (time.time() - start_time) > deadline - 5:
            break
        
        # Time budget per support: more for smaller degree ranges
        span = support[-1] - support[0]
        if span <= 15:
            time_budget = 3.0
        elif span <= 20:
            time_budget = 2.0
        elif span <= 25:
            time_budget = 1.0
        else:
            time_budget = 0.5
        
        time_limit = min(time.time() + time_budget, deadline - 5)
        backtrack_search(support, time_limit)
    
    # Final local search around best if we have time
    if best_poly is not None and (time.time() - start_time) < deadline - 3:
        for _ in range(2000):
            if (time.time() - start_time) > deadline - 3:
                break
            perturbed = list(best_poly)
            # Flip a sign
            nonzero_indices = [i for i, c in enumerate(perturbed) if c != 0]
            if nonzero_indices:
                idx = random.choice(nonzero_indices)
                perturbed[idx] = -perturbed[idx]
                if is_valid(perturbed):
                    score = count_nonzero_square(perturbed)
                    if score < best_score:
                        best_score = score
                        best_poly = list(perturbed)
            
            # Move a term
            if len(best_poly) > 1:
                perturbed = list(best_poly)
                nonzero_indices = [i for i, c in enumerate(perturbed) if c != 0]
                if nonzero_indices:
                    idx = random.choice(nonzero_indices)
                    old_val = perturbed[idx]
                    new_pos = random.randint(0, min(len(perturbed) - 1, 40))
                    if new_pos != idx:
                        perturbed[idx] = 0
                        if new_pos < len(perturbed):
                            perturbed[new_pos] += old_val
                        else:
                            perturbed.extend([0] * (new_pos - len(perturbed) + 1))
                            perturbed[new_pos] = old_val
                        # Check validity
                        nz_count = sum(1 for c in perturbed if c != 0)
                        if nz_count == 12 and perturbed and perturbed[-1] != 0:
                            if all(-64 <= c <= 64 for c in perturbed):
                                score = count_nonzero_square(perturbed)
                                if score < best_score:
                                    best_score = score
                                    best_poly = list(perturbed)
    
    return best_poly if best_poly is not None else [1] * 12
