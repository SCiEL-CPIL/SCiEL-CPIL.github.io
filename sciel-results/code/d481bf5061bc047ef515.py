import time
import random
import numpy as np

def _count_nonzero_square_sparse(exponents, coeffs):
    """Compute nonzero count of P(x)^2 given sparse representation."""
    n = len(exponents)
    max_deg = exponents[-1]
    square = [0] * (2 * max_deg + 1)
    for i in range(n):
        ei = exponents[i]
        ci = coeffs[i]
        for j in range(n):
            ej = exponents[j]
            cj = coeffs[j]
            square[ei + ej] += ci * cj
    return sum(1 for x in square if x != 0)

def _local_refine(exponents, coeffs, best_count, deadline):
    """Try small perturbations to reduce nonzero count in square."""
    n = len(exponents)
    max_deg = exponents[-1]
    current_count = _count_nonzero_square_sparse(exponents, coeffs)
    if current_count >= best_count:
        return exponents, coeffs, current_count
    
    coeff_vals = [-3, -2, -1, 1, 2, 3]
    
    # Try changing one coefficient at a time
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for new_c in coeff_vals:
                if new_c == coeffs[i] or new_c == 0:
                    continue
                old_c = coeffs[i]
                coeffs[i] = new_c
                cnt = _count_nonzero_square_sparse(exponents, coeffs)
                if cnt < current_count:
                    current_count = cnt
                    improved = True
                    break
                elif cnt < best_count:
                    pass  # keep for now but don't update best
                else:
                    coeffs[i] = old_c
        
        if not improved:
            break
    
    # Try shifting one exponent by ±1, ±2 (if within bounds and no collision)
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for delta in [-2, -1, 1, 2]:
                new_e = exponents[i] + delta
                if new_e < 0 or new_e > 40:
                    continue
                if new_e in exponents:
                    continue
                old_e = exponents[i]
                exponents[i] = new_e
                exponents.sort()
                cnt = _count_nonzero_square_sparse(exponents, coeffs)
                if cnt < current_count:
                    current_count = cnt
                    improved = True
                    break
                else:
                    exponents[i] = old_e
                    exponents.sort()
        
        if not improved:
            break
    
    return exponents, coeffs, current_count

def construct():
    start_time = time.time()
    timeout = 55.0
    deadline = start_time + timeout
    
    best_count = float('inf')
    best_exponents = list(range(12))
    best_coeffs = [1]*12
    
    # Phase 1: Brute force signs for 12 consecutive terms (0..11)
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        square = [0]*23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i+j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            best_exponents = list(range(12))

    # Phase 2: Random search for coefficients in {-2, -1, 1, 2} with exponents 0..11
    random.seed(42)
    while time.time() < deadline:
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        exponents = list(range(12))
        cnt = _count_nonzero_square_sparse(exponents, coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = coeffs[:]
            best_exponents = exponents[:]
            if best_count == 12:
                return best_coeffs
        # Local refine if promising
        if cnt <= best_count + 2:
            e_ref, c_ref, cnt_ref = _local_refine(exponents[:], coeffs[:], best_count, deadline)
            if cnt_ref < best_count:
                best_count = cnt_ref
                best_coeffs = c_ref[:]
                best_exponents = e_ref[:]
                if best_count == 12:
                    return best_coeffs

    # Phase 3: Random search with random exponents and coefficients
    while time.time() < deadline:
        exponents = random.sample(range(16), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        cnt = _count_nonzero_square_sparse(exponents, coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = coeffs[:]
            best_exponents = exponents[:]
            if best_count == 12:
                return best_coeffs
        # Local refine
        if cnt <= best_count + 2:
            e_ref, c_ref, cnt_ref = _local_refine(exponents[:], coeffs[:], best_count, deadline)
            if cnt_ref < best_count:
                best_count = cnt_ref
                best_coeffs = c_ref[:]
                best_exponents = e_ref[:]
                if best_count == 12:
                    return best_coeffs

    # Phase 4: Wider random search with more coefficient values and wider exponent range
    while time.time() < deadline:
        exponents = random.sample(range(20), 12)
        exponents.sort()
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        cnt = _count_nonzero_square_sparse(exponents, coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = coeffs[:]
            best_exponents = exponents[:]
            if best_count == 12:
                return best_coeffs
        if cnt <= best_count + 2:
            e_ref, c_ref, cnt_ref = _local_refine(exponents[:], coeffs[:], best_count, deadline)
            if cnt_ref < best_count:
                best_count = cnt_ref
                best_coeffs = c_ref[:]
                best_exponents = e_ref[:]
                if best_count == 12:
                    return best_coeffs

    # Build final result
    max_deg = max(best_exponents)
    full_coeffs = [0] * (max_deg + 1)
    for e, c in zip(best_exponents, best_coeffs):
        full_coeffs[e] = c
    
    return full_coeffs
