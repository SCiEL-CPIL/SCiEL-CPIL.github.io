"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    best_poly = [1] * 12
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square computation using FFT convolution."""
        n = len(coeff_list)
        # Pad to power of 2 for FFT efficiency
        size = 1
        while size < 2 * n - 1:
            size *= 2
        a = np.zeros(size, dtype=np.float64)
        for i, c in enumerate(coeff_list):
            if c != 0:
                a[i] = c
        # FFT convolution
        fa = np.fft.fft(a)
        sq = np.fft.ifft(fa * fa).real
        sq = np.round(sq).astype(np.int64)
        sq = sq[:2 * n - 1]
        return int(np.count_nonzero(sq))
    
    def square_nonzero_count(coeff_list):
        """Fallback square computation."""
        n = len(coeff_list)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeff_list[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeff_list[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sum(1 for v in sq if v != 0)
    
    def trim(poly):
        """Trim trailing zeros, ensure length >= 1."""
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def is_valid(poly):
        """Check if polynomial meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 or c != int(c) for c in poly):
            return False
        return True
    
    # Phase 1: Quick search with ±1 coefficients on structured exponent sets
    exponent_families = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15],
    ]
    
    for exp_set in exponent_families:
        if time.time() > deadline:
            break
        # Try sign patterns (fix first coefficient sign = +1)
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline:
                break
            coeffs = [0] * 41
            coeffs[exp_set[0]] = 1
            for i, s in enumerate(signs):
                coeffs[exp_set[i+1]] = s
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
    
    # Phase 2: Local search - perturb coefficients of best solution
    if time.time() < deadline:
        # Work with the best poly found so far
        work_poly = best_poly[:]
        work_count = best_count
        
        # Try changing individual coefficients to other values
        max_degree = len(work_poly)
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Randomly pick a position to modify
            pos = np.random.randint(0, max_degree)
            old_val = work_poly[pos]
            
            # Try different values
            new_vals = []
            if old_val == 0:
                new_vals = [1, -1, 2, -2, 3, -3]
            elif old_val == 1:
                new_vals = [-1, 2, -2, 0, 3]
            elif old_val == -1:
                new_vals = [1, -2, 2, 0, -3]
            elif old_val == 2:
                new_vals = [1, -1, 0, -2, 3]
            elif old_val == -2:
                new_vals = [-1, 1, 0, 2, -3]
            else:
                new_vals = [1, -1, 2, -2, 0]
            
            for new_val in new_vals:
                if new_val == old_val:
                    continue
                if abs(new_val) > 64:
                    continue
                
                work_poly[pos] = new_val
                
                # Check if still exactly 12 nonzero
                nonzero = sum(1 for c in work_poly if c != 0)
                if nonzero != 12:
                    work_poly[pos] = old_val
                    continue
                
                # Ensure final coefficient is nonzero
                if work_poly[-1] == 0:
                    work_poly[pos] = old_val
                    continue
                
                cnt = square_nonzero_count_fast(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work_poly[:])
                else:
                    work_poly[pos] = old_val
    
    # Phase 3: Try coefficient values of ±2 in structured ways
    if time.time() < deadline:
        # Try patterns with some coefficients = ±2
        for exp_set in exponent_families[:4]:
            if time.time() > deadline:
                break
            
            # Choose 2-3 positions to have coefficient ±2, rest ±1
            n_terms = len(exp_set)
            for n_twos in range(2, 5):
                if time.time() > deadline:
                    break
                # Choose which positions get ±2
                from itertools import combinations
                for two_positions in combinations(range(n_terms), n_twos):
                    if time.time() > deadline:
                        break
                    # Signs for all positions
                    sign_positions = [p for p in range(n_terms) if p not in two_positions]
                    # Fix first sign = +1
                    if sign_positions and sign_positions[0] == 0:
                        rest_signs = sign_positions[1:]
                    else:
                        rest_signs = sign_positions
                    
                    # Try a subset of sign patterns to keep tractable
                    max_patterns = 200
                    patterns = list(product([1, -1], repeat=len(rest_signs)))
                    if len(patterns) > max_patterns:
                        patterns = patterns[:max_patterns]
                    
                    for signs in patterns:
                        if time.time() > deadline:
                            break
                        coeffs = [0] * 41
                        # Set coefficients
                        sign_idx = 0
                        for i, pos in enumerate(range(n_terms)):
                            if i == 0:
                                coeffs[exp_set[0]] = 1
                            elif pos in two_positions:
                                # Determine sign for this position
                                if pos in rest_signs:
                                    s = signs[sign_idx]
                                    sign_idx += 1
                                else:
                                    s = 1
                                coeffs[exp_set[pos]] = 2 * s
                            else:
                                if pos in rest_signs:
                                    s = signs[sign_idx]
                                    sign_idx += 1
                                else:
                                    s = 1
                                coeffs[exp_set[pos]] = s
                        
                        nonzero = sum(1 for c in coeffs if c != 0)
                        if nonzero != 12:
                            continue
                        
                        cnt = square_nonzero_count_fast(coeffs)
                        if cnt < best_count:
                            best_count = cnt
                            best_poly = trim(coeffs[:])
    
    # Phase 4: Final local refinement on best solution
    if time.time() < deadline and is_valid(best_poly):
        work_poly = best_poly[:]
        work_count = best_count
        max_degree = len(work_poly)
        
        # More aggressive local search
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick 1-2 positions to modify
            n_mods = 1 if np.random.random() < 0.7 else 2
            positions = np.random.choice(max_degree, size=n_mods, replace=False)
            
            original = [work_poly[p] for p in positions]
            
            # Generate random new values
            new_vals = []
            for p in positions:
                choices = [1, -1, 2, -2, 3, -3, 0]
                new_vals.append(np.random.choice(choices))
            
            for p, nv in zip(positions, new_vals):
                work_poly[p] = nv
            
            # Validate
            nonzero = sum(1 for c in work_poly if c != 0)
            if nonzero != 12 or work_poly[-1] == 0:
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
                continue
            
            if any(abs(c) > 64 for c in work_poly):
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
                continue
            
            cnt = square_nonzero_count_fast(work_poly)
            if cnt < work_count:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(work_poly[:])
            else:
                for p, ov in zip(positions, original):
                    work_poly[p] = ov
    
    # Final cleanup
    best_poly = trim(best_poly)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
