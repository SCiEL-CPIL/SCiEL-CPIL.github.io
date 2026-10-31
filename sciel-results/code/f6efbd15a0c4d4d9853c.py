import numpy as np
import random

N = 50

def greedy_sidon_with_restarts(N, n_restarts=500):
    best_set = []
    best_len = 0
    
    for _ in range(n_restarts):
        indices = list(range(N))
        random.shuffle(indices)
        
        current_set = []
        sums_used = set()
        
        for idx in indices:
            valid = True
            for j in current_set:
                s = idx + j
                if s in sums_used:
                    valid = False
                    break
            if valid:
                s_ii = idx + idx
                if s_ii in sums_used:
                    valid = False
            
            if valid:
                current_set.append(idx)
                for j in current_set[:-1]:
                    sums_used.add(idx + j)
                sums_used.add(idx + idx)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    return best_set

def construct():
    S = greedy_sidon_with_restarts(N)
    if not S:
        S = [0]
        
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0 / len(S)
        
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
