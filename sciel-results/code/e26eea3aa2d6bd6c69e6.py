import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    n = len(signs)
    # Prefix sums P[0]=0, P[i+1] = P[i] + signs[i]
    P = [0] * (n + 1)
    for i in range(n):
        P[i + 1] = P[i] + signs[i]
    
    max_disc = 0
    
    for d in range(1, n):
        # Max length of AP with step d
        m_max = (n - 1) // d + 1
        # If max possible sum (which is m_max) is <= max_disc, skip
        if m_max <= max_disc:
            continue
            
        for r in range(d):
            if r >= n:
                break
                
            # We need to process the sequence of prefix sums:
            # P[r], P[r+d], P[r+2d], ...
            # Let's find max and min running values to compute max diff.
            
            # The set of indices in P is r, r+d, ...
            # We want max(P[j] - P[i]) for i < j in this set.
            
            min_p = P[r]
            max_p = P[r]
            
            # Iterate k from 1
            idx = r
            while True:
                idx += d
                if idx >= n + 1: # P has length n+1, last index n
                    break
                val = P[idx]
                
                # Potential new max diff ending here
                diff = val - min_p
                if diff > max_disc:
                    max_disc = diff
                    
                diff_neg = max_p - val
                if diff_neg > max_disc:
                    max_disc = diff_neg
                    
                if val < min_p:
                    min_p = val
                if val > max_p:
                    max_p = val
                    
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 256
    rng = random.Random(42)
    
    # Initial candidates
    # 1. Thue-Morse
    signs_tm = [0] * n
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
        
    # 2. Random
    signs_rand = [rng.choice((-1, 1)) for _ in range(n)]
    
    candidates = [signs_tm, signs_rand]
    
    best_signs = candidates[0]
    best_disc = float('inf')
    
    for c in candidates:
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
            
    current_signs = best_signs[:]
    current_disc = best_disc
    
    # Simulated Annealing
    T = 2.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    while time.time() < deadline:
        # Pick random index
        i = rng.randrange(n)
        old_val = current_signs[i]
        current_signs[i] = -old_val
        
        new_disc = _max_ap_discrepancy(current_signs)
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current_signs[:]
        else:
            current_signs[i] = old_val
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_signs
