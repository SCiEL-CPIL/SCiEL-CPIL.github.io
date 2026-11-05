import random
import time
import sys

def count_nonzero_square(poly):
    """
    poly: list of (exponent, coefficient)
    Returns number of non-zero coefficients in poly^2
    """
    # Max degree is 40, so square max degree is 80
    square = [0] * 81
    for i in range(len(poly)):
        e1, c1 = poly[i]
        for j in range(len(poly)):
            e2, c2 = poly[j]
            e = e1 + e2
            if e < 81:
                square[e] += c1 * c2
    return sum(1 for c in square if c != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_poly = [(i, 1) for i in range(12)]  # Seed: 1+x+...+x^11
    best_score = count_nonzero_square(best_poly)
    
    # Random search
    # We want 12 terms.
    # Exponents should be distinct and in [0, 40].
    # Coefficients in [-64, 64]. Small coefficients often lead to cancellations.
    
    def random_poly():
        exponents = random.sample(range(0, 41), 12)
        coefficients = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        # Ensure non-zero final coefficient
        # The problem says "nonzero final coefficient".
        # This usually means the highest degree term is non-zero.
        # Since we pick distinct exponents, the highest exponent will have a non-zero coefficient.
        # We should sort by exponent to be safe, though the square doesn't care about order.
        poly = list(zip(exponents, coefficients))
        poly.sort()
        return poly
    
    # Initial random candidates
    for _ in range(500):
        p = random_poly()
        s = count_nonzero_square(p)
        if s < best_score:
            best_score = s
            best_poly = p
            if best_score <= 13: # Good enough?
                break
        if time.time() > deadline:
            break
            
    # Local search
    # If we found a good one, try to improve it.
    if best_score > 13:
        current_poly = best_poly
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random term to modify
            idx = random.randint(0, 11)
            old_e, old_c = current_poly[idx]
            
            # New exponent
            new_e = random.randint(0, 40)
            # Ensure distinct exponents
            if any(e == new_e for i, (e, c) in enumerate(current_poly) if i != idx):
                continue
                
            # New coefficient
            new_c = random.choice([-3, -2, -1, 1, 2, 3])
            
            # Apply change
            current_poly[idx] = (new_e, new_c)
            current_poly.sort()
            
            new_score = count_nonzero_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly
            else:
                no_improve_count += 1
                # Revert if not improving (simple hill climbing)
                current_poly[idx] = (old_e, old_c)
                current_poly.sort()
                
            if best_score <= 13:
                break

    # Convert best_poly to coefficient list
    # Length <= 41
    # Entries integers in [-64, 64]
    # Exactly 12 nonzero terms
    # Nonzero final coefficient
    
    max_exp = max(e for e, c in best_poly)
    if max_exp > 40:
        # Fallback to seed
        return [1] * 12
        
    res = [0] * (max_exp + 1)
    for e, c in best_poly:
        res[e] = c
        
    # Ensure length <= 41
    if len(res) > 41:
        res = res[:41]
        
    # Ensure nonzero final coefficient
    if res[-1] == 0:
        # This shouldn't happen with our construction, but just in case
        return [1] * 12
        
    return res
