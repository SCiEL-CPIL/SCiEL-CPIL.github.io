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
    
    # Random restarts to find a larger Sidon set
    for _ in range(200):
        S = []
        used_sums = set()
        # Try adding elements in a random order
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            # Check if adding i creates a collision
            # New sums will be i+i and i+j for j in S
            conflict = False
            
            # Check self-sum
            if (i + i) in used_sums:
                conflict = True
            else:
                # Check cross sums with existing elements
                for j in S:
                    if (i + j) in used_sums:
                        conflict = True
                        break
            
            if not conflict:
                S.append(i)
                # Add new sums to used_sums
                used_sums.add(i + i)
                for j in S[:-1]: # S[:-1] is S before adding i? No, S has i now.
                    # Wait, I added i to S. The new sums are i+i and i+j for j in old S.
                    # The loop above checked against old S.
                    # So I should add sums with the elements that were in S BEFORE adding i.
                    # Let's restructure to be safe.
                    pass
                    
        # Let's rewrite the inner logic to be cleaner
        
        # Restart inner logic
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
