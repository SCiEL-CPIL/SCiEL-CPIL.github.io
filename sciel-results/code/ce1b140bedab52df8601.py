import numpy as np
import time
import random

def count_nonzero(poly):
    return np.count_nonzero(poly)

def square_poly(p):
    # p is a list of coefficients
    # Use np.convolve
    return np.convolve(p, p)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 23
    best_poly = [1] * 12
    
    max_deg = 40
    num_terms = 12
    
    # Pre-generate some random candidates?
    # Or just loop.
    
    # To speed up, we can restrict exponents to a smaller range first
    # e.g. 0..20, then expand.
    
    while time.time() < deadline:
        # Random support
        # Ensure we don't pick too many close exponents which create dense squares
        # But dense squares allow cancellation.
        exponents = random.sample(range(max_deg + 1), num_terms)
        exponents.sort()
        
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(num_terms)]
        
        p = [0] * (exponents[-1] + 1)
        for e, c in zip(exponents, coeffs):
            p[e] = c
            
        sq = square_poly(p)
        
        cnt = count_nonzero(sq)
        
        if cnt < best_score:
            best_score = cnt
            best_poly = p[:]
            
            # If we find a very good one, we can try to polish it
            # by swapping coefficients or exponents
            if best_score <= 15:
                # Polish loop
                for _ in range(100):
                    if time.time() > deadline - 5: break
                    # Try changing one coefficient
                    idx = random.randint(0, len(p)-1)
                    if p[idx] == 0: continue
                    old_val = p[idx]
                    new_val = random.choice([-3, -2, -1, 1, 2, 3])
                    if new_val == old_val: continue
                    p[idx] = new_val
                    sq = square_poly(p)
                    cnt = count_nonzero(sq)
                    if cnt < best_score:
                        best_score = cnt
                        best_poly = p[:]
                    else:
                        p[idx] = old_val
                        
    while best_poly and best_poly[-1] == 0:
        best_poly.pop()
        
    return best_poly
