import sys
import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    n1 = 23
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    # Generate subset sums for first half
    size1 = 1 << n1
    arr1 = np.zeros(size1, dtype=np.int64)
    for i in range(n1):
        w = w1[i]
        step = 1 << i
        arr1[step:step*2] = arr1[:step] + w
        
    mask1 = arr1 < L
    sums1 = arr1[mask1]
    masks1 = np.where(mask1)[0]
    
    if time.time() > deadline:
        return [0]
        
    # Generate subset sums for second half
    size2 = 1 << n2
    arr2 = np.zeros(size2, dtype=np.int64)
    for i in range(n2):
        w = w2[i]
        step = 1 << i
        arr2[step:step*2] = arr2[:step] + w
        
    mask2 = arr2 < L
    sums2 = arr2[mask2]
    masks2 = np.where(mask2)[0]
    
    if time.time() > deadline:
        return [0]

    # Sort
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    masks2_sorted = masks2[order2]
    
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    masks1_sorted = masks1[order1]
    
    # For each s1, find largest s2 < L - s1
    targets = L - sums1_sorted
    insert_points = np.searchsorted(sums2_sorted, targets, side='left')
    valid_idx2 = insert_points - 1
    valid_mask = insert_points > 0
    
    if not np.any(valid_mask):
        # No valid pair found, fallback
        return [0]
    
    candidate_sums = sums1_sorted[valid_mask] + sums2_sorted[valid_idx2[valid_mask]]
    # All candidate_sums should be < L by construction
    max_local_idx = np.argmax(candidate_sums)
    best_sum = candidate_sums[max_local_idx]
    best_idx1 = np.where(valid_mask)[0][max_local_idx]
    best_idx2 = valid_idx2[valid_mask][max_local_idx]
    
    if time.time() > deadline:
        pass
    
    m1 = masks1_sorted[best_idx1]
    m2 = masks2_sorted[best_idx2]
    
    indices = []
    for k in range(n1):
        if m1 & (1 << k):
            indices.append(k)
            
    for k in range(n2):
        if m2 & (1 << k):
            indices.append(n1 + k)
            
    if not indices:
        indices = [0]
        
    return indices
