import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

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
    
    def generate_sums(weights_list):
        size = len(weights_list)
        total_masks = 1 << size
        sums = [0] * total_masks
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                s = sums[mask] + w
                if s < L:
                    sums[bit | mask] = s
                else:
                    sums[bit | mask] = L  # mark as invalid
        valid_pairs = [(sums[m], m) for m in range(total_masks) if sums[m] < L]
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    if time.time() > deadline:
        return [0]
    
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    right_sums = [p[0] for p in right_pairs]
    right_masks = [p[1] for p in right_pairs]
    
    if time.time() > deadline:
        return [0]
    
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        target = L - ls
        idx = bisect.bisect_left(right_sums, target) - 1
        
        if idx >= 0:
            rs = right_sums[idx]
            rm = right_masks[idx]
            deficit = L - ls - rs
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
                if deficit == 1:
                    break
    
    if best_left_mask == -1:
        return [0]
    
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
