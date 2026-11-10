import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s budget, 10s safety margin
    
    best_nonzero = 144  # baseline: 12*12 with no cancellation
    best_poly = None
    
    def count_nonzero(poly_coeffs):
        """Count nonzero entries in a coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def compute_square_nonzero(coeffs):
        """Given coefficient list (ascending degree), compute square and count nonzero terms."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    sq[i + j] += val
                else:
                    sq[i + j] += 2 * val
        # Trim trailing zeros
        while len(sq) > 1 and sq[-1] == 0:
            sq.pop()
        return count_nonzero(sq)
    
    def make_poly(exponents, coefficients, max_deg):
        """Create coefficient list from exponents and coefficients."""
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] = c
        return coeffs
    
    # Strategy 1: Arithmetic progression exponents with small integer coefficients
    # P has 12 terms at exponents a, a+d, a+2d, ..., a+11d
    # Coefficients from small set
    
    coeff_choices = [-2, -1, 1, 2]
    
    for d in range(1, 15):  # step size
        for a in range(0, 5):  # start offset
            exponents = [a + i * d for i in range(12)]
            max_deg = max(exponents)
            
            # Try all coefficient combinations - too many (4^12 = 16M), so use smarter search
            # Instead, try symmetric patterns and small variations
            
            # Try all-1 first
            coeffs = [1] * 12
            poly = make_poly(exponents, coeffs, max_deg)
            nz = compute_square_nonzero(poly)
            if nz < best_nonzero:
                best_nonzero = nz
                best_poly = poly[:]
            
            # Try alternating signs
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            poly = make_poly(exponents, coeffs, max_deg)
            nz = compute_square_nonzero(poly)
            if nz < best_nonzero:
                best_nonzero = nz
                best_poly = poly[:]
            
            # Try palindromic coefficients
            for c1 in coeff_choices:
                for c2 in coeff_choices:
                    for c3 in coeff_choices:
                        for c4 in coeff_choices:
                            for c5 in coeff_choices:
                                for c6 in coeff_choices:
                                    coeffs = [c1, c2, c3, c4, c5, c6, c6, c5, c4, c3, c2, c1]
                                    poly = make_poly(exponents, coeffs, max_deg)
                                    nz = compute_square_nonzero(poly)
                                    if nz < best_nonzero:
                                        best_nonzero = nz
                                        best_poly = poly[:]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: P = A + x^k * A, where A is 6-term
    # A has 6 terms, P has 12 terms (no overlap)
    for k in range(20, 60):
        # A: 6 terms at positions 0, d1, d2, d3, d4, d5
        for d in range(1, 10):
            exps_A = [0, d, 2*d, 3*d, 4*d, 5*d]
            # Ensure no overlap: k > 5*d
            if k <= 5 * d:
                continue
            
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    for c3 in [-2, -1, 1, 2]:
                        for c4 in [-2, -1, 1, 2]:
                            for c5 in [-2, -1, 1, 2]:
                                for c6 in [-2, -1, 1, 2]:
                                    coeffs_A = [c1, c2, c3, c4, c5, c6]
                                    # Build P
                                    max_deg = max(exps_A) + k + max(exps_A)
                                    poly = [0] * (max_deg + 1)
                                    for e, c in zip(exps_A, coeffs_A):
                                        poly[e] = c
                                        poly[e + k] = c
                                    
                                    # Verify exactly 12 nonzero terms
                                    nz_poly = sum(1 for x in poly if x != 0)
                                    if nz_poly != 12:
                                        continue
                                    
                                    nz_sq = compute_square_nonzero(poly)
                                    if nz_sq < best_nonzero:
                                        best_nonzero = nz_sq
                                        best_poly = poly[:]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 3: Clustered exponent patterns
    # Group exponents into clusters to create more sum collisions
    cluster_patterns = [
        # Two clusters of 6
        lambda: list(range(0, 6)) + list(range(10, 16)),
        # Three clusters of 4
        lambda: list(range(0, 4)) + list(range(8, 12)) + list(range(16, 20)),
        # Mixed
        lambda: [0,1,2,3,10,11,12,13,20,21,22,23],
    ]
    
    for pattern_fn in cluster_patterns:
        exponents = pattern_fn()
        if len(exponents) != 12 or len(set(exponents)) != 12:
            continue
        max_deg = max(exponents)
        
        # Try all-ones
        coeffs = [1] * 12
        poly = make_poly(exponents, coeffs, max_deg)
        nz = compute_square_nonzero(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = poly[:]
        
        # Try alternating
        coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
        poly = make_poly(exponents, coeffs, max_deg)
        nz = compute_square_nonzero(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = poly[:]
        
        if time.time() > deadline:
            break
    
    # Strategy 4: Random search with hill climbing
    import random
    random.seed(42)
    
    def random_poly():
        """Generate a random 12-term polynomial."""
        # Choose 12 distinct exponents in [0, 40]
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
        max_deg = max(exps)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            poly[e] = c
        return poly
    
    def mutate_poly(poly):
        """Mutate a polynomial: change one coefficient or one exponent."""
        new_poly = poly[:]
        nz_indices = [i for i, c in enumerate(new_poly) if c != 0]
        if not nz_indices:
            return new_poly
        
        op = random.random()
        if op < 0.4:
            # Change a coefficient
            idx = random.choice(nz_indices)
            new_poly[idx] = random.choice([-3,-2,-1,1,2,3])
            if new_poly[idx] == 0:
                # Try to keep 12 terms
                new_poly[idx] = random.choice([-2,-1,1,2])
        elif op < 0.7:
            # Shift one term
            idx = random.choice(nz_indices)
            shift = random.choice([-2,-1,1,2])
            new_idx = idx + shift
            if 0 <= new_idx < len(new_poly) and new_poly[new_idx] == 0:
                new_poly[new_idx] = new_poly[idx]
                new_poly[idx] = 0
            elif new_idx >= len(new_poly):
                # Extend
                new_poly.extend([0] * (new_idx - len(new_poly) + 1))
                new_poly[new_idx] = new_poly[idx]
                new_poly[idx] = 0
        else:
            # Swap two coefficients
            if len(nz_indices) >= 2:
                i, j = random.sample(nz_indices, 2)
                new_poly[i], new_poly[j] = new_poly[j], new_poly[i]
        
        # Ensure exactly 12 nonzero terms
        nz_count = sum(1 for c in new_poly if c != 0)
        if nz_count != 12:
            # Try to fix
            if nz_count > 12:
                # Remove some
                for idx in random.sample(nz_indices, nz_count - 12):
                    if new_poly[idx] != 0:
                        new_poly[idx] = 0
            elif nz_count < 12:
                # Add some
                zero_indices = [i for i, c in enumerate(new_poly) if c == 0]
                for idx in random.sample(zero_indices, 12 - nz_count):
                    new_poly[idx] = random.choice([-2,-1,1,2])
        
        # Trim trailing zeros but keep at least length 1
        while len(new_poly) > 1 and new_poly[-1] == 0:
            new_poly.pop()
        
        # Verify
        if sum(1 for c in new_poly if c != 0) != 12:
            return poly
        
        return new_poly
    
    # Hill climbing
    current = random_poly()
    current_nz = compute_square_nonzero(current)
    if current_nz < best_nonzero:
        best_nonzero = current_nz
        best_poly = current[:]
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        candidate = mutate_poly(current)
        cand_nz = compute_square_nonzero(candidate)
        
        if cand_nz < current_nz:
            current = candidate
            current_nz = cand_nz
            no_improve = 0
            if current_nz < best_nonzero:
                best_nonzero = current_nz
                best_poly = current[:]
        else:
            no_improve += 1
        
        # Occasionally restart
        if no_improve > 200 and random.random() < 0.01:
            current = random_poly()
            current_nz = compute_square_nonzero(current)
            no_improve = 0
    
    # Return best found
    if best_poly is not None:
        # Trim to ensure valid format
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        # Ensure length <= 41
        if len(best_poly) > 41:
            best_poly = best_poly[:41]
            while len(best_poly) > 1 and best_poly[-1] == 0:
                best_poly.pop()
        # Ensure nonzero final coefficient
        if best_poly[-1] == 0:
            # Find last nonzero
            for i in range(len(best_poly)-1, -1, -1):
                if best_poly[i] != 0:
                    best_poly = best_poly[:i+1]
                    break
        # Ensure entries in [-64, 64]
        best_poly = [max(-64, min(64, c)) for c in best_poly]
        # Ensure exactly 12 nonzero terms
        nz = sum(1 for c in best_poly if c != 0)
        if nz == 12 and len(best_poly) <= 41 and best_poly[-1] != 0:
            return best_poly
    
    # Fallback: return a valid 12-term polynomial
    return [1] * 12
