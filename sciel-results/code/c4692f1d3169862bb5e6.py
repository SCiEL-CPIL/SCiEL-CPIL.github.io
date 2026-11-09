import random
import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    # Initial candidate: 12 terms, all 1s, spread out
    # Try to find a better one
    best_poly = [0]*41
    for i in range(12):
        best_poly[i] = 1
    best_score = 23 # Max possible for 12 terms is 23
    
    # Helper to count nonzeros in square
    def count_nonzero_square(poly):
        # poly is list of coeffs
        n = len(poly)
        # Convolution
        square = [0]*(2*n-1)
        for i in range(n):
            if poly[i] == 0: continue
            for j in range(n):
                if poly[j] == 0: continue
                square[i+j] += poly[i]*poly[j]
        
        count = 0
        for c in square:
            if c != 0:
                count += 1
        return count
    
    # Initial score
    current_poly = best_poly[:]
    current_score = count_nonzero_square(current_poly)
    
    # Local Search
    # State: list of (exponent, coeff)
    # We represent poly as list of length 41 for ease, but track active terms
    
    active_terms = [(i, current_poly[i]) for i in range(12)]
    
    while time.time() - start_time < timeout:
        # Random move
        move_type = random.choice(['change_coeff', 'move_term', 'sign_flip', 'swap_terms'])
        
        if move_type == 'change_coeff':
            idx = random.randint(0, len(active_terms)-1)
            old_exp, old_coeff = active_terms[idx]
            new_coeff = random.randint(-64, 64)
            if new_coeff == 0:
                continue # Must remain nonzero
            new_terms = active_terms[:]
            new_terms[idx] = (old_exp, new_coeff)
            
            # Reconstruct poly
            new_poly = [0]*41
            for e, c in new_terms:
                if e < 41:
                    new_poly[e] = c
            
            # Check validity (12 terms, distinct exponents)
            exps = [t[0] for t in new_terms]
            if len(set(exps)) != 12:
                continue
            if max(exps) >= 41:
                continue
            if new_poly[-1] == 0: # Nonzero final coefficient requirement? 
                # "nonzero final coefficient" usually means the highest degree term is nonzero.
                # Since we have 12 terms, the max exponent term must be nonzero.
                pass
            
            new_score = count_nonzero_square(new_poly)
            
            if new_score < current_score:
                current_poly = new_poly
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            elif new_score == current_score and random.random() < 0.1:
                current_poly = new_poly
                current_score = new_score
                
        elif move_type == 'move_term':
            idx = random.randint(0, len(active_terms)-1)
            old_exp, old_coeff = active_terms[idx]
            new_exp = random.randint(0, 40)
            
            # Check if new_exp is already used
            if any(t[0] == new_exp for i, t in enumerate(active_terms) if i != idx):
                continue
                
            new_terms = active_terms[:]
            new_terms[idx] = (new_exp, old_coeff)
            
            new_poly = [0]*41
            for e, c in new_terms:
                if e < 41:
                    new_poly[e] = c
            
            new_score = count_nonzero_square(new_poly)
            
            if new_score < current_score:
                current_poly = new_poly
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            elif new_score == current_score and random.random() < 0.1:
                current_poly = new_poly
                current_score = new_score
                
        elif move_type == 'sign_flip':
            idx = random.randint(0, len(active_terms)-1)
            old_exp, old_coeff = active_terms[idx]
            new_coeff = -old_coeff
            
            new_terms = active_terms[:]
            new_terms[idx] = (old_exp, new_coeff)
            
            new_poly = [0]*41
            for e, c in new_terms:
                if e < 41:
                    new_poly[e] = c
            
            new_score = count_nonzero_square(new_poly)
            
            if new_score < current_score:
                current_poly = new_poly
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            elif new_score == current_score and random.random() < 0.1:
                current_poly = new_poly
                current_score = new_score

        # Update active_terms from current_poly to keep them in sync
        active_terms = [(i, current_poly[i]) for i in range(41) if current_poly[i] != 0]
        
        # If we hit a very low score, we can stop early
        if best_score <= 12:
            break

    return best_poly

