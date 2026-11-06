import numpy as np
import time
from math import gcd
from functools import reduce

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 49))
    weights = [L // d for d in denominators]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def gen_sums(weights):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights):
            bit = 1 << i
            current_len = len(sums)
            for j in range(current_len):
                sums.append(sums[j] + w)
                masks.append(masks[j] | bit)
        return sums, masks

    sums1, masks1 = gen_sums(w1)
    sums2, masks2 = gen_sums(w2)
    
    # Convert to NumPy
    A = np.array(sums1, dtype=np.int64)
    M1 = np.array(masks1, dtype=np.int64)
    
    B = np.array(sums2, dtype=np.int64)
    M2 = np.array(masks2, dtype=np.int64)
    
    # Sort B and M2
    idxs = np.argsort(B)
    B_sorted = B[idxs]
    M2_sorted = M2[idxs]
    
    # Vectorized search
    # We want max A[i] + B[j] < L
    # For each A[i], we want max B[j] < L - A[i]
    targets = L - A
    # searchsorted returns index where target would be inserted to maintain order
    # side='left' means if target is present, it returns the first index.
    # We want the element strictly less than target.
    # So we take index - 1.
    pos = np.searchsorted(B_sorted, targets, side='left')
    
    # If pos[i] == 0, there is no B[j] < targets[i] (since B[0] is min, if targets[i] <= B[0], no valid j)
    # But B[0] is 0. targets[i] = L - A[i].
    # If A[i] >= L, targets[i] <= 0. Since B >= 0, no B < targets.
    # If A[i] < L, targets[i] > 0. Since B[0]=0, and 0 < targets[i] (if targets[i]>0), then pos[i] >= 1.
    # So pos[i] > 0 is the valid condition.
    
    valid_mask = pos > 0
    
    if not np.any(valid_mask):
        # Fallback, shouldn't happen
        return [0]
        
    # We only care about valid indices
    # We need to find the pair (i, j) that maximizes A[i] + B[j] subject to A[i] + B[j] < L
    # For each valid i, the best j is pos[i] - 1.
    best_j_indices = pos[valid_mask] - 1
    valid_i_indices = np.where(valid_mask)[0]
    
    A_valid = A[valid_i_indices]
    B_selected = B_sorted[best_j_indices]
    
    Total_Sums = A_valid + B_selected
    
    # We want strictly less than L.
    # The construction ensures Total_Sums < L.
    # We want to minimize L - Total_Sums, which is equivalent to maximizing Total_Sums.
    
    max_sum_idx = np.argmax(Total_Sums)
    
    best_i = valid_i_indices[max_sum_idx]
    best_j = best_j_indices[max_sum_idx]
    
    mask1 = M1[best_i]
    mask2 = M2_sorted[best_j]
    
    # Reconstruct
    final_indices = []
    for i in range(mid):
        if mask1 & (1 << i):
            final_indices.append(i)
            
    for i in range(len(w2)):
        if mask2 & (1 << i):
            final_indices.append(mid + i)
            
    if not final_indices:
        final_indices = [0]
        
    return final_indices
