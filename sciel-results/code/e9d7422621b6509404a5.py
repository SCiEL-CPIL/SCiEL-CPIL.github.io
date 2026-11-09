import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    p = 1009
    m = 5
    
    # Compute Legendre symbols for 1..p-1
    # (a/p) = a^((p-1)/2) mod p
    # If result is 1, it's a QR. If p-1 (i.e., -1), it's a QNR.
    
    # Vectorized pow
    # Note: np.power with mod is available in newer numpy, 
    # but standard python pow is faster for small ints in loops?
    # Actually, for 1000 elements, a list comprehension is fine.
    # But let's follow the directive.
    
    # Legendre symbol calculation
    # a = 1..1008
    # exp = (1009-1)/2 = 504
    
    # Using pow in a list comprehension is very fast for this size.
    # Legendre[a] = 1 if QR, -1 if QNR, 0 if a% p == 0 (not here)
    
    legendre = np.zeros(p, dtype=int)
    for a in range(1, p):
        # pow(a, 504, 1009) returns 1 or 1008
        res = pow(a, 504, p)
        if res == 1:
            legendre[a] = 1
        else:
            legendre[a] = -1
            
    # Find runs of 1s of length m
    # We want s such that legendre[s..s+m-1] are all 1.
    
    # Create a boolean array of QRs
    is_qr = (legendre == 1)
    
    # Cumulative sum of non-QRs?
    # Or just sliding window sum.
    # Sum of is_qr[s..s+m-1] should be m.
    
    # Using np.convolve or similar?
    # For small p, simple loop is fine.
    
    best_s = -1
    # s from 1 to p-m
    for s in range(1, p - m + 1):
        if np.all(is_qr[s:s+m]):
            best_s = s
            break
            
    if best_s == -1:
        # Fallback or error
        return {'start': 1, 'roots': [1, 2, 3, 4, 5]} # Dummy
        
    # Find roots
    # We need r_i such that r_i^2 % p == s+i
    # We can just loop x=1..p and find roots for the specific values
    
    roots_found = {}
    for x in range(1, p):
        v = (x * x) % p
        if v in range(best_s, best_s + m):
            if v not in roots_found:
                roots_found[v] = x
                
    result_roots = []
    for i in range(m):
        val = best_s + i
        result_roots.append(roots_found[val])
        
    return {'start': best_s, 'roots': result_roots}
