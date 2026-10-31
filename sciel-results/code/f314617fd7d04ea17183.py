import numpy as np
import random

N = 50

def construct():
    best_S = []
    # Random restarts greedy approach
    for _ in range(2000):
        S = []
        sums = set()
        indices = list(range(N))
        random.shuffle(indices)
        
        for i in indices:
            # Check if adding i maintains Sidon property
            # New sums: i+i and i+j for all j in S
            new_sums = [i + i]
            valid = True
            for j in S:
                s = i + j
                new_sums.append(s)
            
            # Check for collisions with existing sums
            for s in new_sums:
                if s in sums:
                    valid = False
                    break
            
            if valid:
                S.append(i)
                for s in new_sums:
                    sums.add(s)
        
        if len(S) > len(best_S):
            best_S = S[:]
            # Optional: break if we hit theoretical max, but we don't know it for sure
            # sqrt(50) is ~7.07. Max is likely 7 or 8.
            # If we find 8, we can stop early? Let's run all to be safe, or stop if len==8.
            if len(S) >= 8:
                # It is unlikely to go much higher for N=50. 
                # But let's continue a bit more to be sure or just break.
                # Breaking on 8 might be safe if 8 is the max.
                # Let's just collect best.
                pass

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
