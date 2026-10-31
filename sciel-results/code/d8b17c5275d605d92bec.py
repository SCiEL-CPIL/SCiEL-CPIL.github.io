import numpy as np
import random
import time

N = 50

def construct():
    best_S = []
    best_size = 0
    start_time = time.time()
    max_time = 110  # Leave buffer for safety

    # We want to find the largest Sidon set in {0, ..., N-1}
    # A Sidon set S has the property that all sums i+j (i <= j) are distinct.
    # For uniform weights on a Sidon set of size k, the ratio is k/2.
    # Thus, maximizing k maximizes the score.
    
    # Strategy: Random restarts with greedy construction.
    for restart in range(2000):
        if time.time() - start_time > max_time:
            break
            
        S = []
        used_sums = set()
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            # Check if i can be added to S
            # New sums: i+i, i+j for all j in S
            new_sums = [i + i]
            for j in S:
                new_sums.append(i + j)
                
            # If any new sum is already in used_sums, we have a collision
            if any(s in used_sums for s in new_sums):
                continue
                
            S.append(i)
            used_sums.update(new_sums)
            
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
            # If we find a reasonably large set (e.g., 7 for N=50), we can stop early
            # Max possible is around sqrt(N). For N=50, 7 is likely max.
            if best_size >= 7:
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
