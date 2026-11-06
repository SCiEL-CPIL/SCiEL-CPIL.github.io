import numpy as np
import time
import random

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin
    
    # Seed with a known good candidate if found, else random
    best_poly = None
    best_count = 100
    
    # We need exactly 12 nonzero terms.
    # Let's try coefficients of -1, 0, 1.
    # To ensure exactly 12 nonzero terms, we can pick 12 positions and assign signs.
    
    # Try a few structured candidates first
    # 1. The 6-term one extended? No.
    # 2. Random search
    
    # To make the search effective, we can fix the first term to 1 and search for the rest.
    # But 12 terms is a lot.
    # Let's try a smaller search: 12 terms, coeff +-1.
    
    # Heuristic: Polynomials with symmetry often have sparse squares.
    
    # Let's try to find a polynomial P such that P^2 has few terms.
    # We will iterate over random sets of 12 exponents and signs.
    
    max_iter = 20000
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        # Generate random 12-term polynomial
        # Choose 12 distinct exponents from 0 to 40
        exps = random.sample(range(41), 12)
        exps.sort()
        # Assign random signs
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        
        # Construct polynomial array
        # Max degree is max(exps)*2
        max_deg = max(exps) * 2
        p = np.zeros(max_deg + 1, dtype=int)
        for e, c in zip(exps, coeffs):
            p[e] = c
            
        # Compute square
        s = np.convolve(p, p, mode='full')
        
        # Count non-zeros
        count = np.count_nonzero(s)
        
        if count < best_count:
            best_count = count
            best_poly = p.copy()
            
            if best_count <= 9:
                break
                
    if best_poly is None:
        # Fallback to seed
        return [1] * 12
        
    # Convert to list, trim trailing zeros
    res = best_poly.tolist()
    while res and res[-1] == 0:
        res.pop()
        
    # Ensure length <= 41 and nonzero final coeff
    if len(res) > 41:
        # This shouldn't happen if max_deg < 41
        res = res[:41]
        while res and res[-1] == 0:
            res.pop()
            
    if not res:
        return [1] * 12
        
    return res
