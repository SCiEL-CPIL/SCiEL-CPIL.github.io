import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_poly = None
    best_score = float('inf')  # Number of non-zero terms in square
    
    # Helper to count non-zero terms in the square
    def get_square_sparsity(exponents, coeffs):
        # exponents: list of 12 ints
        # coeffs: list of 12 ints
        # Returns number of non-zero coefficients in P^2
        
        # Max degree is 80. Use a list of size 81.
        # This is faster than dict for small fixed range.
        square = [0] * 81
        
        n = 12
        # i <= j
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(i, n):
                ej = exponents[j]
                cj = coeffs[j]
                deg = ei + ej
                if i == j:
                    square[deg] += ci * cj
                else:
                    square[deg] += 2 * ci * cj
        
        # Count non-zeros
        count = 0
        for v in square:
            if v != 0:
                count += 1
        return count

    def validate_and_update(exponents, coeffs):
        nonlocal best_poly, best_score
        
        # Check constraints
        if len(exponents) != 12 or len(coeffs) != 12:
            return False
        if len(set(exponents)) != 12:
            return False
        if any(e < 0 or e > 40 for e in exponents):
            return False
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return False
        
        # Calculate sparsity
        sparsity = get_square_sparsity(exponents, coeffs)
        
        if sparsity < best_score:
            best_score = sparsity
            # Construct the polynomial list
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            # Ensure length <= 41 (max_deg <= 40 so len <= 41)
            # Ensure non-zero final coefficient
            if poly[-1] != 0:
                best_poly = poly
            return True
        return False

    # Phase 1: Structured Initialization
    # Try APs with various coefficient patterns
    # Focus on small coefficient values to allow cancellation
    
    random.seed(42)
    
    # Candidate exponents
    exp_candidates = []
    # APs
    for d in range(1, 6):
        for a in range(0, 41 - 11*d + 1):
            exp_candidates.append([a + i*d for i in range(12)])
    # Random sets
    for _ in range(2000):
        exps = random.sample(range(41), 12)
        exps.sort()
        exp_candidates.append(exps)
    # Clustered sets (to encourage collisions in exponents sums)
    for _ in range(2000):
        base = random.randint(0, 30)
        exps = []
        for i in range(12):
            e = base + random.randint(0, 10)
            if e not in exps and e <= 40:
                exps.append(e)
        while len(exps) < 12:
            e = random.randint(0, 40)
            if e not in exps:
                exps.append(e)
        exps.sort()
        exp_candidates.append(exps)
        
    # Candidate coefficients
    # Small integers are better for cancellation
    coeff_vals = [-3, -2, -1, 1, 2, 3]
    
    # Test a subset of exps with random coeffs
    # We can't test all 4000 * huge combos.
    # Let's test each exp candidate with a few random coeff patterns.
    
    for exps in exp_candidates:
        if time.time() > deadline - 10:
            break
        # 5 random patterns per exponent set
        for _ in range(5):
            coeffs = [random.choice(coeff_vals) for _ in range(12)]
            validate_and_update(exps, coeffs)
            
    # Ensure we have a best_poly
    if best_poly is None:
        # Fallback: simple valid poly
        exps = list(range(12))
        coeffs = [1]*12
        validate_and_update(exps, coeffs)
        if best_poly is None:
            return [1]*12

    # Phase 2: Local Search
    # Extract current best
    current_exp = [i for i, c in enumerate(best_poly) if c != 0]
    current_coef = [c for c in best_poly if c != 0]
    current_score = best_score
    
    # Sort by exponent to keep consistent
    pairs = sorted(zip(current_exp, current_coef))
    current_exp = [p[0] for p in pairs]
    current_coef = [p[1] for p in pairs]
    
    # Simulated Annealing / Hill Climbing
    # Temperature starts high, cools down.
    # We accept worse moves with probability exp(-delta/T) to escape local minima.
    
    temp = 2.0
    cooling = 0.9999
    
    for iteration in range(200000):
        if time.time() > deadline - 5:
            break
        
        # Generate neighbor
        exp_copy = list(current_exp)
        coef_copy = list(current_coef)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Change exponent
            idx = random.randint(0, 11)
            shift = random.randint(-5, 5)
            if shift == 0: continue
            new_e = exp_copy[idx] + shift
            if 0 <= new_e <= 40:
                # Check collision
                if new_e not in exp_copy:
                    exp_copy[idx] = new_e
                    exp_copy.sort()
                    # Re-align coeffs with new sorted exponents?
                    # This is tricky. If we sort exponents, we must sort coeffs accordingly.
                    # Let's just keep unsorted lists for the neighbor generation, 
                    # and sort only for validation if needed, or just use a dict/set check.
                    # Actually, get_square_sparsity doesn't require sorted exponents.
                    # But validate_and_update constructs the poly.
                    # Let's just check validity before calling validate.
                    if len(set(exp_copy)) == 12:
                        pass # Valid
                    else:
                        continue
                else:
                    continue
            else:
                continue
        elif move_type < 0.9:
            # Change coefficient value
            idx = random.randint(0, 11)
            new_c = random.choice(coeff_vals)
            if new_c != 0:
                coef_copy[idx] = new_c
            else:
                continue
        else:
            # Change coefficient sign
            idx = random.randint(0, 11)
            coef_copy[idx] = -coef_copy[idx]
        
        # Quick validation
        if len(set(exp_copy)) != 12:
            continue
        if any(e < 0 or e > 40 for e in exp_copy):
            continue
        if any(c == 0 or abs(c) > 64 for c in coef_copy):
            continue
            
        new_score = get_square_sparsity(exp_copy, coef_copy)
        
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_exp = exp_copy
            current_coef = coef_copy
            current_score = new_score
            validate_and_update(current_exp, current_coef)
        else:
            # Worse or same
            # Accept with probability
            if random.random() < pow(2.71828, -delta / temp):
                current_exp = exp_copy
                current_coef = coef_copy
                current_score = new_score
                # Do not update global best if worse, but keep local state
                # Actually, if we accept a worse move, we might wander.
                # We should only update global best if it's an improvement.
        
        # Cool down
        temp *= cooling
        if temp < 0.1:
            temp = 0.1 # Minimum temperature to allow some exploration

    # Final check
    if best_poly is None:
        return [1]*12
        
    # Ensure returned list is valid
    if best_poly[-1] == 0:
        for i in range(len(best_poly)-1, -1, -1):
            if best_poly[i] != 0:
                best_poly = best_poly[:i+1]
                break
        if not best_poly:
            return [1]
            
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        return [1]*12
        
    return best_poly
