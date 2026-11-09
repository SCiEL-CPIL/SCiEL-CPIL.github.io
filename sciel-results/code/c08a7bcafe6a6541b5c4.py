import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_count = 144  # worst case: all 144 terms nonzero
    best_poly = None
    
    # Fast square computation using dictionary-based approach
    def compute_square(exps, coeffs):
        """Compute square of polynomial given exponents and coefficients."""
        sq = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei, ci = exps[i], coeffs[i]
            for j in range(i, n):
                ej, cj = exps[j], coeffs[j]
                deg = ei + ej
                val = ci * cj
                if i == j:
                    sq[deg] += val
                else:
                    sq[deg] += 2 * val
        return sq
    
    def count_nz(sq):
        return sum(1 for v in sq.values() if v != 0)
    
    def to_coeff_list(exps, coeffs):
        if not exps:
            return [0]
        max_deg = max(exps)
        result = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            result[e] = c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    # Structural approach: P = A * B where A and B have 6 terms each
    # Search over pairs of 6-term polynomials whose product has exactly 12 nonzero terms
    # Then P^2 = A^2 * B^2
    
    # Pre-generate candidate 6-term polynomials with small exponents
    random.seed(12345)
    
    # Generate diverse 6-term base polynomials
    def gen_6term_poly():
        """Generate a random 6-term polynomial with exponents in [0, 20]."""
        exps = sorted(random.sample(range(21), 6))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        return exps, coeffs
    
    def poly_product(exps_a, coeffs_a, exps_b, coeffs_b):
        """Multiply two polynomials, return (exps, coeffs) with exactly 12 nonzero terms if possible."""
        prod = defaultdict(int)
        for i in range(len(exps_a)):
            for j in range(len(exps_b)):
                deg = exps_a[i] + exps_b[j]
                prod[deg] += coeffs_a[i] * coeffs_b[j]
        # Filter to nonzero terms
        result_exps = []
        result_coeffs = []
        for deg in sorted(prod.keys()):
            if prod[deg] != 0:
                result_exps.append(deg)
                result_coeffs.append(prod[deg])
        return result_exps, result_coeffs
    
    def poly_square_count(exps, coeffs):
        """Count nonzero terms in square of polynomial."""
        sq = compute_square(exps, coeffs)
        return count_nz(sq)
    
    # Phase 1: Find P = A*B with exactly 12 nonzero terms and sparse square
    # Try many random pairs of 6-term polynomials
    
    candidate_pool = []
    
    # Generate structured 6-term polynomials
    structured_6term = []
    
    # Arithmetic progression based
    for start in range(0, 5):
        for d in range(1, 5):
            exps = [start + i * d for i in range(6)]
            if max(exps) <= 20:
                for coeff_pattern in [[1,1,1,1,1,1], [1,-1,1,-1,1,-1], [1,1,-1,-1,1,1], [1,-1,-1,1,1,-1]]:
                    structured_6term.append((tuple(exps), tuple(coeff_pattern)))
    
    # Random 6-term polynomials
    for _ in range(500):
        exps, coeffs = gen_6term_poly()
        structured_6term.append((tuple(exps), tuple(coeffs)))
    
    # Remove duplicates
    structured_6term = list(set(structured_6term))
    
    # Try all pairs (sampled for speed)
    n_struct = len(structured_6term)
    pair_attempts = 0
    max_pairs = 50000
    
    for i in range(n_struct):
        if time.time() > deadline:
            break
        exps_a, coeffs_a = structured_6term[i]
        for j in range(i, n_struct):
            if time.time() > deadline:
                break
            pair_attempts += 1
            if pair_attempts > max_pairs:
                break
            
            exps_b, coeffs_b = structured_6term[j]
            
            # Compute product
            prod_exps, prod_coeffs = poly_product(list(exps_a), list(coeffs_a), list(exps_b), list(coeffs_b))
            
            # Check: exactly 12 nonzero terms, degree <= 40
            if len(prod_exps) != 12 or max(prod_exps) > 40:
                continue
            
            # Check coefficient bounds
            if any(abs(c) > 64 for c in prod_coeffs):
                continue
            
            # Count nonzero terms in square
            sq_count = poly_square_count(prod_exps, prod_coeffs)
            
            if sq_count < best_count:
                best_count = sq_count
                best_poly = (list(prod_exps), list(prod_coeffs))
                if best_count <= 15:
                    break
        
        if time.time() > deadline:
            break
    
    # Phase 2: Local search / simulated annealing on the best found
    if best_poly is not None:
        exps, coeffs = best_poly
        current_count = best_count
        
        # Simulated annealing on coefficients
        temp = 10.0
        cooling = 0.95
        coeff_range = [-4, -3, -2, -1, 1, 2, 3, 4]
        
        for sa_iter in range(20000):
            if time.time() > deadline:
                break
            
            # Pick random coefficient to change
            idx = random.randint(0, 11)
            old_coeff = coeffs[idx]
            new_coeff = random.choice(coeff_range)
            if new_coeff == old_coeff:
                continue
            if abs(new_coeff) > 64:
                continue
            
            coeffs[idx] = new_coeff
            
            # Recompute square count
            new_count = poly_square_count(exps, coeffs)
            
            # Accept if better or with probability exp(-delta/temp)
            delta = new_count - current_count
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.1)):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_poly = (list(exps), list(coeffs))
            else:
                coeffs[idx] = old_coeff
            
            temp *= cooling
        
        # Local search: try changing individual coefficients
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_c = coeffs[i]
                for new_c in coeff_range:
                    if new_c == old_c or abs(new_c) > 64:
                        continue
                    coeffs[i] = new_c
                    new_count = poly_square_count(exps, coeffs)
                    if new_count < best_count:
                        best_count = new_count
                        best_poly = (list(exps), list(coeffs))
                        improved = True
                    else:
                        coeffs[i] = old_c
    
    # Phase 3: Also try direct search on 12-term polynomials with structured exponents
    # Focus on exponent sets that promote collision in the square
    
    # Exponent sets where differences repeat (promotes collision)
    special_exp_sets = []
    
    # Sets with many repeated differences
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            if d1 == d2:
                continue
            # Alternating differences
            exps = [0]
            for k in range(11):
                if k % 2 == 0:
                    exps.append(exps[-1] + d1)
                else:
                    exps.append(exps[-1] + d2)
            if max(exps) <= 40 and len(set(exps)) == 12:
                special_exp_sets.append(tuple(exps))
    
    # Symmetric sets around a center
    for center in range(10, 30):
        for spread in range(1, 10):
            exps = sorted(set([center - spread*i for i in range(6)] + [center + spread*i for i in range(6)]))
            if len(exps) == 12 and max(exps) <= 40 and min(exps) >= 0:
                special_exp_sets.append(tuple(exps))
    
    special_exp_sets = list(set(special_exp_sets))
    
    for exp_set in special_exp_sets:
        if time.time() > deadline:
            break
        
        # Try coefficient patterns that promote cancellation
        # Pattern: symmetric coefficients
        for pattern_type in range(4):
            if time.time() > deadline:
                break
            
            coeffs = []
            for k in range(12):
                if pattern_type == 0:
                    # Alternating signs
                    coeffs.append(1 if k % 2 == 0 else -1)
                elif pattern_type == 1:
                    # First half +, second half -
                    coeffs.append(1 if k < 6 else -1)
                elif pattern_type == 2:
                    # Symmetric: same from both ends
                    coeffs.append(1 if k < 6 else -1)
                else:
                    # Random
                    coeffs.append(random.choice([-1, 1]))
            
            sq_count = poly_square_count(list(exp_set), coeffs)
            if sq_count < best_count:
                best_count = sq_count
                best_poly = (list(exp_set), list(coeffs))
    
    # Fallback
    if best_poly is None:
        best_poly = (list(range(12)), [1]*12)
        best_count = poly_square_count(*best_poly)
    
    # Convert to coefficient list
    result = to_coeff_list(best_poly[0], best_poly[1])
    
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
    
    # Ensure nonzero final coefficient
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Ensure entries in [-64, 64]
    result = [max(-64, min(64, c)) for c in result]
    
    # Verify exactly 12 nonzero terms
    nonzero = [(i, c) for i, c in enumerate(result) if c != 0]
    if len(nonzero) != 12:
        # Fallback to a valid 12-term polynomial
        fallback_exps = list(range(12))
        fallback_coeffs = [1]*12
        result = to_coeff_list(fallback_exps, fallback_coeffs)
    
    return result
