# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Precompute conflict bitmasks
    conflict_mask = [0] * n
    for i in range(n):
        ai = numbers[i]
        for j in range(i + 1, n):
            aj = numbers[j]
            g = gcd(ai, aj)
            if ai * aj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def can_add(idx, selected_mask):
        return (conflict_mask[idx] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(selected_mask):
        s = 0.0
        for i in range(n):
            if (selected_mask >> i) & 1:
                s += 1.0 / numbers[i]
        return s
    
    def greedy_construct(order):
        selected_mask = 0
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask |= (1 << idx)
        return selected_mask
    
    def local_search(selected_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding any single element
            for idx in range(n):
                if not ((selected_mask >> idx) & 1):
                    if can_add(idx, selected_mask):
                        selected_mask |= (1 << idx)
                        improved = True
                        break
            
            if time.time() >= time_limit:
                break
            
            # Try 1-for-1 swap
            if not improved:
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if not ((selected_mask >> i) & 1):
                        continue
                    # Try removing i and adding j
                    mask_without_i = selected_mask & ~(1 << i)
                    for j in range(n):
                        if ((selected_mask >> j) & 1):
                            continue
                        if can_add(j, mask_without_i):
                            new_mask = mask_without_i | (1 << j)
                            # Compare sums
                            old_sum = reciprocal_sum_from_mask(selected_mask)
                            new_sum = reciprocal_sum_from_mask(new_mask)
                            if new_sum > old_sum + 1e-15:
                                selected_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 1-for-2 or 2-for-1 moves (add two, remove one, etc.)
            if not improved and time.time() < time_limit:
                # Try adding two elements simultaneously
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if ((selected_mask >> i) & 1):
                        continue
                    for j in range(i + 1, n):
                        if ((selected_mask >> j) & 1):
                            continue
                        if can_add(i, selected_mask) and can_add(j, selected_mask) and can_add(i, selected_mask | (1 << j)) and can_add(j, selected_mask | (1 << i)):
                            new_mask = selected_mask | (1 << i) | (1 << j)
                            old_sum = reciprocal_sum_from_mask(selected_mask)
                            new_sum = reciprocal_sum_from_mask(new_mask)
                            if new_sum > old_sum + 1e-15:
                                selected_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
        
        return selected_mask
    
    best_mask = 0
    best_sum = 0.0
    
    # Deterministic first pass: sort by descending reciprocal (ascending number)
    order_sorted = list(range(n))  # already in order of increasing number = decreasing reciprocal
    mask = greedy_construct(order_sorted)
    mask = local_search(mask, deadline - 1.0)
    s = reciprocal_sum_from_mask(mask)
    if s > best_sum:
        best_sum = s
        best_mask = mask
    
    # Randomized restarts
    trial = 0
    while time.time() < deadline - 1.0:
        trial += 1
        order = list(range(n))
        random.shuffle(order)
        mask = greedy_construct(order)
        mask = local_search(mask, deadline - 0.5)
        s = reciprocal_sum_from_mask(mask)
        if s > best_sum:
            best_sum = s
            best_mask = mask
        
        if trial >= 200:
            break
    
    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(numbers[i])
    
    if not result:
        result = [251]
    
    return result
# EVOLVE-BLOCK-END
