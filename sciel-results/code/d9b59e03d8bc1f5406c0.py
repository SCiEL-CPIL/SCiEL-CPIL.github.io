import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill (or 90s if that's the limit, but 55s is safe for 60s/90s)
    
    # We focus on the densest structure: 12 consecutive terms.
    # Exponents: 0, 1, ..., 11
    # This maximizes the number of overlapping terms in the square (convolution).
    # The square will have degree 22.
    
    best_count = 100  # Initialize high
    best_coeffs = [1] * 12  # Fallback
    
    def evaluate(coeffs):
        """
        coeffs: list of 12 integers.
        Returns number of non-zero terms in the square.
        """
        # Square of sum c_i x^i is sum_{i,j} c_i c_j x^{i+j}
        # Since exponents are 0..11, square exponents are 0..22
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            if ci == 0: continue
            for j in range(12):
                cj = coeffs[j]
                if cj == 0: continue
                sq[i + j] += ci * cj
        return sum(1 for x in sq if x != 0)

    def update_best(coeffs, count):
        nonlocal best_count, best_coeffs
        if count < best_count:
            best_count = count
            best_coeffs = coeffs[:]
            return True
        return False

    # Phase 1: Brute force all 2^12 sign patterns for coeffs in {-1, 1}
    # This is fast: 4096 * 144 ops ~ 0.6M ops.
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = evaluate(coeffs)
        if update_best(coeffs, count):
            if best_count == 12:  # Theoretical min for 12 terms? Maybe lower? 
                # Actually, if square has 12 terms, that's very sparse.
                # But 13 is the known best from inspiration.
                pass

    # Phase 2: Local Search / Hill Climbing with coefficients in {-2, -1, 1, 2}
    # Start from the best sign pattern found in Phase 1
    # And also try random starts.
    
    random.seed(42)
    vals = [-2, -1, 1, 2]
    
    # Helper for local search
    def local_search(start_coeffs, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        coeffs = start_coeffs[:]
        current_count = evaluate(coeffs)
        update_best(coeffs, current_count)
        
        no_improve = 0
        while time.time() - t0 < time_budget:
            # Pick a random index to modify
            idx = random.randrange(12)
            new_val = random.choice(vals)
            
            if new_val == coeffs[idx]:
                continue
                
            old_val = coeffs[idx]
            coeffs[idx] = new_val
            new_count = evaluate(coeffs)
            
            if new_count < current_count:
                current_count = new_count
                update_best(coeffs, current_count)
                no_improve = 0
            else:
                # Accept with some probability? Or just reject? 
                # Hill climbing is fast. Let's try simple hill climbing first.
                # If we want to escape local minima, we can occasionally jump.
                coeffs[idx] = old_val
                no_improve += 1
                if no_improve > 100:
                    # Random jump: change 2-3 random coeffs
                    for _ in range(random.randint(2, 3)):
                        j = random.randrange(12)
                        coeffs[j] = random.choice(vals)
                    current_count = evaluate(coeffs)
                    update_best(coeffs, current_count)
                    no_improve = 0
                    
    # Run local search from the best Phase 1 result
    local_search(best_coeffs, time_budget=3.0)
    
    # Run local search from several random starts
    while time.time() < deadline:
        random_coeffs = [random.choice(vals) for _ in range(12)]
        local_search(random_coeffs, time_budget=1.0)
        
        # Early exit if we find a very sparse square
        if best_count <= 13:
             # We can continue to see if we can get 12 or less, but 13 is a big jump from 15.
             # Let's run a bit longer to be sure.
             pass

    # Construct the final polynomial list
    # Exponents are 0..11
    result = [0] * 41
    for i in range(12):
        result[i] = best_coeffs[i]
        
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
        
    if not result:
        result = [1]
        
    return result
