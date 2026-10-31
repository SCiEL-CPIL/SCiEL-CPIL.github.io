import numpy as np
import random

N = 50

def construct():
    best_S = []
    best_size = 0
    
    # Increase restarts significantly to ensure we find the largest possible Sidon set
    # 10000 restarts should be well within time limits
    for _ in range(10000):
        S = []
        used_sums = set()
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            new_sums = [i + i]
            for j in S:
                new_sums.append(i + j)
                
            if any(s in used_sums for s in new_sums):
                continue
                
            S.append(i)
            used_sums.update(new_sums)
            
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
            # Optional: early exit if we reach a known upper bound?
            # Upper bound for Sidon set in {0..N-1} is approx sqrt(N) + ...
            # For N=50, sqrt(50) ~ 7.07.
            # If we find 8, that's great. If we find 7, we might stop if we're confident 8 is impossible?
            # But we don't know if 8 is possible without searching.
            # However, if we find 8, we can stop immediately as it's likely the max.
            if best_size >= 8:
                break

    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
