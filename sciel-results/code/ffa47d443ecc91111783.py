import numpy as np
import time
import random

def construct():
    N = 51975
    start_time = time.time()
    time_limit = 110.0
    
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    coverage = np.zeros(N, dtype=np.int32)
    classes = {}
    
    # Greedy Initialization
    for m in divisors:
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            best_r = 0
        else:
            counts = np.bincount(uncovered_indices % m, minlength=m)
            best_r = int(np.argmax(counts))
            
        classes[m] = best_r
        coverage[best_r::m] += 1
        
    best_score = int(np.sum(coverage == 0))
    best_classes = classes.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    # Pre-allocate masks
    mask1 = np.zeros(N, dtype=bool)
    mask2 = np.zeros(N, dtype=bool)
    
    while time.time() - start_time < time_limit:
        m = random.choice(divisors)
        r_old = classes[m]
        r_new = random.randint(0, m - 1)
        
        if r_new == r_old:
            continue
            
        mask1[:] = 0
        mask2[:] = 0
        mask1[r_old::m] = True
        mask2[r_new::m] = True
        
        diff_old = mask1 & ~mask2
        diff_new = mask2 & ~mask1
        
        delta = 0
        if np.any(diff_old):
            delta += int(np.sum(coverage[diff_old] == 1))
        if np.any(diff_new):
            delta -= int(np.sum(coverage[diff_new] == 0))
            
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            coverage[diff_old] -= 1
            coverage[diff_new] += 1
            classes[m] = r_new
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_classes = classes.copy()
                
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor to avoid underflow issues, or restart?
            # Restarting SA might help escape local minima.
            # But we have a time limit. Just keep going.
            
    return [[m, r] for m, r in best_classes.items()]
