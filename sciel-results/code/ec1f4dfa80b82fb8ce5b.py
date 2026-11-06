import numpy as np
import random
import time

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Phase 1: Greedy Set Cover
    cover_count = np.zeros(base, dtype=np.int32)
    uncovered = np.ones(base, dtype=bool)
    
    classes = []
    used_moduli = set()
    available_moduli = list(divisors)
    
    while np.any(uncovered) and available_moduli:
        if time.time() - start_time > time_limit:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in available_moduli:
            # Pruning: max possible gain is base // m
            if base // m <= best_gain:
                continue
                
            max_gain_for_m = 0
            best_r_for_m = 0
            
            # Find best r for this m
            # To speed up, we can check r in a random order? No, must find max.
            # We can skip if m is very large and best_gain is already high?
            # base // m is the upper bound.
            
            for r in range(m):
                gain = np.sum(uncovered[r::m])
                if gain > max_gain_for_m:
                    max_gain_for_m = gain
                    best_r_for_m = r
            
            if max_gain_for_m > best_gain:
                best_gain = max_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None or best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available_moduli.remove(best_m)
        
        # Update cover_count and uncovered
        cover_count[best_r::best_m] += 1
        uncovered[best_r::best_m] = False
        
    # Phase 2: Local Search
    improved = True
    # Limit iterations to avoid infinite loops or time outs
    max_passes = 5
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() - start_time > time_limit:
            break
            
        for i in range(len(classes)):
            m, r_old = classes[i]
            
            best_r_new = r_old
            best_delta = 0
            
            # Try all new residues
            for r_new in range(m):
                if r_new == r_old:
                    continue
                    
                old_slice = slice(r_old, base, m)
                new_slice = slice(r_new, base, m)
                
                lost = np.sum(cover_count[old_slice] == 1)
                gained = np.sum(cover_count[new_slice] == 0)
                
                delta = lost - gained
                
                if delta < best_delta:
                    best_delta = delta
                    best_r_new = r_new
            
            if best_r_new != r_old:
                old_slice = slice(r_old, base, m)
                new_slice = slice(best_r_new, base, m)
                
                cover_count[old_slice] -= 1
                cover_count[new_slice] += 1
                classes[i][1] = best_r_new
                improved = True
                
    return classes
