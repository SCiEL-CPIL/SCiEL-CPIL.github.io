import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Known good candidate from inspirations: 11 nonzero terms in square
    # This corresponds to score 23/11 = 2.09
    # We try to find better (fewer terms)
    
    best_poly = None
    best_score = 23  # Max possible terms
    
    # Strategy: Search for P = Q * R where Q, R have 6 terms
    # Or just random search for 12-term polynomials
    
    # Candidate 1: The one from inspiration
    # P(x) = 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^10 - x^11 ?
    # Let's check a few known sparse square polynomials
    
    candidates = [
        # [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        # [1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0]
        [1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0],
        # [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
    ]
    
    def count_nonzero(poly):
        return sum(1 for c in poly if c != 0)
        
    def square(poly):
        n = len(poly)
        res = [0] * (2*n - 1)
        for i in range(n):
            if poly[i] == 0: continue
            for j in range(n):
                if poly[j] == 0: continue
                res[i+j] += poly[i] * poly[j]
        # Trim trailing zeros
        while len(res) > 0 and res[-1] == 0:
            res.pop()
        return res

    # Evaluate candidates
    for c in candidates:
        sq = square(c)
        nnz = count_nonzero(sq)
        if nnz < best_score:
            best_score = nnz
            best_poly = c
            
    # Random search
    random.seed(42)
    while time.time() < deadline:
        # Generate random 12-term polynomial
        # Choose 12 distinct indices
        indices = random.sample(range(40), 12)
        indices.sort()
        coeffs = [0] * 40
        for i in indices:
            coeffs[i] = random.choice([-1, 1])
            
        # Ensure length <= 41 and nonzero final
        # Trim to max index + 1
        max_idx = max(indices)
        poly = coeffs[:max_idx+1]
        
        if len(poly) > 41 or poly[-1] == 0:
            continue
            
        if count_nonzero(poly) != 12:
            continue
            
        sq = square(poly)
        nnz = count_nonzero(sq)
        
        if nnz < best_score:
            best_score = nnz
            best_poly = poly
            
        if best_score <= 2: # Theoretical minimum for non-monomial is likely > 2
            break
            
    if best_poly is None:
        best_poly = [1] * 12
        
    return best_poly
