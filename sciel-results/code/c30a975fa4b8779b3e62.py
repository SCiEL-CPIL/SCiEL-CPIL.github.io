import time
import bisect
from math import lcm

_best_result = None

def construct():
    global _best_result
    if _best_result is not None:
        return _best_result

    start_time = time.time()
    deadline = start_time + 45.0
    
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
    
    # Generate all subset sums for left half
    left_sums = [0] * (1 << left_size)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Generate all subset sums for right half
    right_sums = [0] * (1 << right_size)
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
            
    # Sort right sums with their masks
    right_pairs = [(right_sums[mask], mask) for mask in range(1 << right_size)]
    right_pairs.sort()
    right_sum_values = [p[0] for p in right_pairs]
    
    best_deficit = L
    best_left_mask = 0
    best_right_mask = 0
    
    for mask in range(1 << left_size):
        if time.time() > deadline:
            break
            
        ls = left_sums[mask]
        if ls >= L:
            continue
        limit = L - ls
        idx = bisect.bisect_left(right_sum_values, limit) - 1
        if idx >= 0:
            rs = right_sum_values[idx]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = mask
                best_right_mask = right_pairs[idx][1]
                
    result_indices = []
    for i in range(left_size):
        if best_left_mask & (1 << i):
            result_indices.append(i)
    for i in range(right_size):
        if best_right_mask & (1 << i):
            result_indices.append(left_size + i)
            
    if not result_indices:
        result_indices = [0]
        
    _best_result = result_indices
    return _best_result
