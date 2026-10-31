import numpy as np
import random

N = 50

def construct():
    # Find a large Sidon set in {0, ..., N-1}
    # A Sidon set has distinct pairwise sums i+j (i <= j)
    # This minimizes the peak of the autoconvolution for uniform weights,
    # maximizing the ratio R = ||h||_2^2 / (||h||_1 * max(h*h))
    
    best_S = []
    best_size = 0
    
    # Random restarts to find a larger Sidon set with local search improvement
    for restart in range(300):
        S = []
        used_sums = set()
        candidates = list(range(N))
        random.shuffle(candidates)
        
        # Greedy construction
        for i in candidates:
            new_sums = [i + i]
            for j in S:
                new_sums.append(i + j)
                
            if any(s in used_sums for s in new_sums):
                continue
                
            S.append(i)
            used_sums.update(new_sums)
        
        # Local search: try to improve by swapping one element for another
        # to potentially allow adding more elements
        S_set = set(S)
        for _ in range(50):  # limited iterations
            improved = False
            # Try removing each element and see if we can add a better one
            for remove_val in list(S_set):
                S_without = [x for x in S_set if x != remove_val]
                S_without_set = set(S_without)
                
                # Compute all sums from S_without
                sums_wo = set()
                for a in S_without:
                    sums_wo.add(a + a)
                    for b in S_without:
                        if a < b:
                            sums_wo.add(a + b)
                
                # Try to find an element not in S that can be added
                for add_val in range(N):
                    if add_val in S_without_set:
                        continue
                    new_sums = [add_val + add_val]
                    for j in S_without:
                        new_sums.append(add_val + j)
                    if any(s in sums_wo for s in new_sums):
                        continue
                    # We found a valid swap
                    S_set.discard(remove_val)
                    S_set.add(add_val)
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                break
        
        S = list(S_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
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
