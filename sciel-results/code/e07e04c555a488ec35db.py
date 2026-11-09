import numpy as np
import time

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    uncovered = np.ones(base, dtype=bool)
    available_moduli = divisors.copy()
    solution = []
    
    start_time = time.time()
    
    while available_moduli and (time.time() - start_time) < 110:
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in available_moduli:
            counts = np.bincount(unc_idx % m, minlength=m)
            max_gain = np.max(counts)
            
            if max_gain > best_gain:
                best_gain = int(max_gain)
                best_m = m
                best_r = int(np.argmax(counts))
        
        if best_gain <= 0:
            break
            
        solution.append([best_m, best_r])
        uncovered[best_r::best_m] = False
        available_moduli.remove(best_m)
        
    return solution
