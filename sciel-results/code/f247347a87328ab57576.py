import time
import numpy as np
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    denominators = list(range(2, 49))
    n = len(denominators)

    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    half = n // 2
    left_size = half
    right_size = n - half

    left_weights = weights[:left_size]
    right_weights = weights[left_size:]

    def generate_sums_with_masks(weights_list):
        size = len(weights_list)
        total_masks = 1 << size
        sums = [0] * total_masks
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + w
        # Filter and return (sum, mask) pairs where sum < L
        valid_pairs = []
        for mask in range(total_masks):
            s = sums[mask]
            if s < L:
                valid_pairs.append((s, mask))
        return valid_pairs

    left_pairs = generate_sums_with_masks(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums_with_masks(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    if time.time() > deadline:
        return [0]
    
    right_sums = [rs for rs, _ in right_pairs]
    
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        target = L - ls
        idx = np.searchsorted(right_sums, target, side='left') - 1
        
        if idx >= 0:
            rs, rm = right_pairs[idx]
            total = ls + rs
            if total < L:
                deficit = L - total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_mask = lm
                    best_right_mask = rm
                    if deficit == 1:
                        break
    
    if best_left_mask == -1:
        best_left_mask = 0
        best_right_mask = 0
        best_deficit = L

    result_indices = []
    for k in range(left_size):
        if best_left_mask & (1 << k):
            result_indices.append(k)
            
    for k in range(right_size):
        if best_right_mask & (1 << k):
            result_indices.append(left_size + k)
            
    if not result_indices:
        result_indices = [0]
        
    return result_indices
