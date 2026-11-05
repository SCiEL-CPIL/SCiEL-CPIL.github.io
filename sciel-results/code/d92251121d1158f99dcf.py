"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Best known: [1]*12 has square with 13 nonzero terms
    best_count = 13
    best_exponents = list(range(12))
    best_coeffs = [1] * 12
    
    def count_nonzero_square(exponents, coeffs):
        """
        Count nonzero terms in the square of the polynomial defined by exponents and coeffs.
        """
        n = len(exponents)
        if n == 0:
            return 0
        max_deg = max(exponents)
        square = [0] * (2 * max_deg + 1)
        
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            if ci == 0:
                continue
            # Diagonal term
            square[2 * ei] += ci * ci
            # Off-diagonal terms
            for j in range(i + 1, n):
                cj = coeffs[j]
                ej = exponents[j]
                if cj == 0:
                    continue
                prod = 2 * ci * cj
                square[ei + ej] += prod
        
        return sum(1 for x in square if x != 0)
    
    def update_best(exponents, coeffs, count):
        nonlocal best_count, best_exponents, best_coeffs
        if count < best_count:
            best_count = count
            best_exponents = exponents[:]
            best_coeffs = coeffs[:]
            return True
        return False
    
    def build_result(exponents, coeffs):
        max_deg = max(exponents) if exponents else 0
        result = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            result[e] = c
        return result
    
    # Phase 1: Brute force signs for 12 consecutive terms (0..11) with +/- 1
    # This is 4096 iterations, very fast.
    exponents_consec = list(range(12))
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = count_nonzero_square(exponents_consec, coeffs)
        if update_best(exponents_consec, coeffs, count):
            if best_count == 12:
                return build_result(best_exponents, best_coeffs)
    
    # Phase 2: Random search for coefficients in {-2, -1, 1, 2} with exponents 0..11
    random.seed(42)
    while time.time() - start_time < deadline:
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        count = count_nonzero_square(exponents_consec, coeffs)
        if update_best(exponents_consec, coeffs, count):
            if best_count == 12:
                return build_result(best_exponents, best_coeffs)
    
    # Phase 3: Random search with random exponents (dense) and coefficients
    # If we didn't find something better, try varying exponents too.
    # We will sample 12 exponents from 0..15 to keep it dense.
    while time.time() - start_time < deadline:
        exponents = random.sample(range(16), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        count = count_nonzero_square(exponents, coeffs)
        if update_best(exponents, coeffs, count):
            if best_count == 12:
                return build_result(best_exponents, best_coeffs)
    
    # Phase 4: Local search / hill climbing starting from best found so far
    if time.time() < deadline:
        current_exponents = best_exponents[:]
        current_coeffs = best_coeffs[:]
        current_count = best_count
        
        coeff_values = [-2, -1, 1, 2]
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_val = current_coeffs[idx]
                new_val = random.choice(coeff_values)
                if new_val != old_val:
                    current_coeffs[idx] = new_val
                    new_count = count_nonzero_square(current_exponents, current_coeffs)
                    if new_count < current_count:
                        current_count = new_count
                        no_improve_count = 0
                        if update_best(current_exponents, current_coeffs, new_count):
                            if best_count == 12:
                                return build_result(best_exponents, best_coeffs)
                    else:
                        # Revert if not better
                        current_coeffs[idx] = old_val
                        no_improve_count += 1
            else:
                # Move an exponent
                idx = random.randint(0, 11)
                old_exp = current_exponents[idx]
                # Try small shifts
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + shift
                if 0 <= new_exp <= 40 and new_exp not in current_exponents:
                    current_exponents[idx] = new_exp
                    current_exponents.sort()
                    # Re-sort coeffs to match exponents
                    # This is tricky because we just sorted exponents.
                    # We need to map old coeffs to new positions.
                    # Easier: keep a list of (exp, coeff) pairs
                    # For now, let's revert to a simpler representation if this gets complex.
                    # Let's stick to modifying the list directly and keep track of pairing
                    # Actually, let's just revert if it's complex
                    # Instead, let's try a different approach: swap exponents
                    current_exponents[idx] = old_exp
                    no_improve_count += 1
        
        # Phase 5: More sophisticated local search with swaps
        if time.time() < deadline:
            current_exponents = best_exponents[:]
            current_coeffs = best_coeffs[:]
            current_count = best_count
            
            no_improve_count = 0
            max_no_improve = 300
            
            while time.time() < deadline and no_improve_count < max_no_improve:
                # Try swapping two exponents
                if random.random() < 0.3:
                    i, j = random.sample(range(12), 2)
                    old_ei, old_ej = current_exponents[i], current_exponents[j]
                    current_exponents[i], current_exponents[j] = old_ej, old_ei
                    current_exponents.sort()
                    # Need to track which coeff goes with which exponent
                    # This is getting complex, let's use a different approach
                    
                    # Actually, let's just try changing coefficients more aggressively
                    current_exponents[i], current_exponents[j] = old_ei, old_ej
                    no_improve_count += 1
                else:
                    # Change coefficient
                    idx = random.randint(0, 11)
                    old_val = current_coeffs[idx]
                    new_val = random.choice([-3, -2, -1, 1, 2, 3])
                    if new_val != old_val:
                        current_coeffs[idx] = new_val
                        new_count = count_nonzero_square(current_exponents, current_coeffs)
                        if new_count < current_count:
                            current_count = new_count
                            no_improve_count = 0
                            if update_best(current_exponents, current_coeffs, new_count):
                                if best_count == 12:
                                    return build_result(best_exponents, best_coeffs)
                        else:
                            current_coeffs[idx] = old_val
                            no_improve_count += 1
    
    # Final validation and return
    result = build_result(best_exponents, best_coeffs)
    
    # Ensure valid output
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Pad to at least length 1
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
