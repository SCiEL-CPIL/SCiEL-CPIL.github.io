import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if int(np.count_nonzero(coeffs)) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        arr = np.array(poly_list, dtype=np.int64)
        if not validate(arr):
            return
        score = evaluate(arr)
        if score < best_score:
            best_score = score
            best_poly = arr.copy()

    random.seed(12345)

    # Phase 1: Structured search over exponents and signs
    # We search for 12 exponents from a range [0, 30]
    # We use a greedy/local approach to find good exponent sets first
    
    # Initialize with a known decent candidate or random start
    # Let's try a few specific structured patterns first
    patterns = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # Consecutive
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], # Even
        [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047], # Powers of 2 (too big, max degree 40)
        [0, 1, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37], # Mixed
    ]
    
    # Since max degree is 40, exponents must be in [0, 40]
    # Let's do a local search on exponents and signs
    
    # Start with a random valid polynomial
    curr_exps = sorted(random.sample(range(41), 12))
    curr_signs = [1] * 12
    curr_vals = [1] * 12
    
    def get_poly_from_params(exps, signs, vals):
        poly = [0] * 41
        for e, s, v in zip(exps, signs, vals):
            if 0 <= e < 41:
                poly[e] = s * v
        # Trim
        while len(poly) > 0 and poly[-1] == 0:
            poly.pop()
        if len(poly) == 0:
            return None
        # Ensure exactly 12 non-zeros (this logic is fragile if collisions happen in poly itself, 
        # but exps are unique so no collisions in P(x))
        return poly

    # Evaluate current
    curr_poly = get_poly_from_params(curr_exps, curr_signs, curr_vals)
    if curr_poly:
        try_poly(curr_poly)
        
    curr_score = best_score

    no_improve = 0
    max_no_improve = 50000

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Pick a random modification type
        mod_type = random.choice(['exp', 'sign', 'val', 'exp_sign', 'swap'])
        
        new_exps = curr_exps.copy()
        new_signs = curr_signs.copy()
        new_vals = curr_vals.copy()
        
        idx = random.randint(0, 11)
        
        if mod_type == 'exp':
            # Change one exponent
            old_e = new_exps[idx]
            # Try moving to a random nearby or random position
            if random.random() < 0.5:
                new_e = max(0, min(40, old_e + random.choice([-2, -1, 1, 2])))
            else:
                new_e = random.randint(0, 40)
            
            # Check for collision in exponents
            if new_e in new_exps and new_exps[idx] != new_e:
                # If collision, skip or resolve? 
                # P(x) must have 12 non-zero terms. If two terms have same exp, they add up.
                # If they cancel, term becomes 0 (bad, <12 terms).
                # If they add, term becomes sum (still 1 term, but 11 terms total in P? No, 12 terms in P means 12 non-zero coeffs.
                # If we have 12 exponents, and two are same, we have 11 distinct positions.
                # The constraint is "exactly 12 nonzero terms" in the *input* polynomial.
                # So we must maintain 12 distinct non-zero coefficients.
                # If we change exp to match another, the coefficient at that exp becomes sum.
                # If sum is 0, we have 11 non-zero terms. Invalid.
                # If sum is non-zero, we have 11 non-zero terms? No, 12 terms were placed, 2 merged into 1. Total 11 non-zero. Invalid.
                # So we must avoid exponent collisions.
                no_improve += 1
                continue
            
            new_exps[idx] = new_e
            new_exps.sort()
            
        elif mod_type == 'sign':
            new_signs[idx] *= -1
            
        elif mod_type == 'val':
            # Change magnitude
            old_v = new_vals[idx]
            # Try small values
            candidates = [1, 2, 3, 4, 5, 10, 20, 40, 64]
            new_v = random.choice(candidates)
            # Ensure it doesn't become 0
            if new_v == 0:
                new_v = 1
            new_vals[idx] = new_v
            
        elif mod_type == 'exp_sign':
            # Change exponent and flip sign
            old_e = new_exps[idx]
            if random.random() < 0.5:
                new_e = max(0, min(40, old_e + random.choice([-1, 1])))
            else:
                new_e = random.randint(0, 40)
                
            if new_e in new_exps and new_exps[idx] != new_e:
                no_improve += 1
                continue
                
            new_exps[idx] = new_e
            new_exps.sort()
            new_signs[idx] *= -1
            
        elif mod_type == 'swap':
            # Swap two exponents? No, that doesn't change the set.
            # Swap values?
            j = random.randint(0, 11)
            new_vals[idx], new_vals[j] = new_vals[j], new_vals[idx]
            # Or swap signs?
            new_signs[idx], new_signs[j] = new_signs[j], new_signs[idx]

        new_poly = get_poly_from_params(new_exps, new_signs, new_vals)
        if new_poly is None:
            no_improve += 1
            continue
            
        new_score = evaluate(new_poly)
        
        if new_score < curr_score:
            curr_score = new_score
            no_improve = 0
            curr_exps = new_exps
            curr_signs = new_signs
            curr_vals = new_vals
            try_poly(new_poly)
            if best_score <= 13:
                break
        else:
            no_improve += 1

    if best_poly is None:
        best_poly = np.array([1] * 12, dtype=np.int64)

    # Trim trailing zeros
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
