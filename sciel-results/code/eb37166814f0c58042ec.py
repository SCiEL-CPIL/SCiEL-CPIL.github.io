import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
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
    
    # Build the best subset found
    in_subset = [False] * n
    for i in range(left_size):
        if best_left_mask & (1 << i):
            in_subset[i] = True
    for i in range(right_size):
        if best_right_mask & (1 << i):
            in_subset[left_size + i] = True
    
    current_sum = sum(weights[i] for i in range(n) if in_subset[i])
    
    # Local search polish: try 1-swap and 1-for-2-swap improvements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-swap: remove one in-subset element, add one out-subset element
        in_indices = [i for i in range(n) if in_subset[i]]
        out_indices = [i for i in range(n) if not in_subset[i]]
        
        for i in in_indices:
            if time.time() > deadline:
                break
            for j in out_indices:
                new_sum = current_sum - weights[i] + weights[j]
                if 0 < new_sum < L:
                    new_deficit = L - new_sum
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        current_sum = new_sum
                        in_subset[i] = False
                        in_subset[j] = True
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 1-for-2-swap: remove one in-subset, add two out-subset
        for i in in_indices:
            if time.time() > deadline:
                break
            for j in out_indices:
                if time.time() > deadline:
                    break
                for k in out_indices:
                    if k == j:
                        continue
                    new_sum = current_sum - weights[i] + weights[j] + weights[k]
                    if 0 < new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            current_sum = new_sum
                            in_subset[i] = False
                            in_subset[j] = True
                            in_subset[k] = True
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-1-swap: remove two in-subset, add one out-subset
        for i in range(len(in_indices)):
            if time.time() > deadline:
                break
            for k in range(i + 1, len(in_indices)):
                if time.time() > deadline:
                    break
                for j in out_indices:
                    new_sum = current_sum - weights[in_indices[i]] - weights[in_indices[k]] + weights[j]
                    if 0 < new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            current_sum = new_sum
                            in_subset[in_indices[i]] = False
                            in_subset[in_indices[k]] = False
                            in_subset[j] = True
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    result_indices = [i for i in range(n) if in_subset[i]]
    
    if not result_indices:
        result_indices = [0]
    
    return result_indices
