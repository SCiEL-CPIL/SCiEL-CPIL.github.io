import time
from math import lcm

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    N = 22
    # Calculate LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i
    
    # Sort indices by weight descending (larger weights first)
    # Larger weights are less likely to create subset sum conflicts
    indices_by_weight = sorted(range(1, N + 1), key=lambda i: weights[i], reverse=True)
    
    best_count = 0
    best_result = []
    
    def dfs(pos, included, sums, count):
        nonlocal best_count, best_result
        if time.time() - start_time > deadline:
            return
        
        # Pruning: even if we include all remaining, we can't beat best
        if count + (len(indices_by_weight) - pos) <= best_count:
            return
        
        # Update best
        if count > best_count:
            best_count = count
            best_result = list(included)
        
        if pos == len(indices_by_weight):
            return
        
        idx = indices_by_weight[pos]
        w = weights[idx]
        
        # Option 1: Include idx (try this first to maximize count)
        # Check if adding w creates a conflict with existing subset sums
        new_sums = set()
        valid = True
        for s in sums:
            if s + w in sums:
                valid = False
                break
            new_sums.add(s)
            new_sums.add(s + w)
        
        if valid and len(new_sums) == 2 * len(sums):
            included.append(idx)
            dfs(pos + 1, included, new_sums, count + 1)
            included.pop()
        
        # Option 2: Exclude idx
        dfs(pos + 1, included, sums, count)
    
    dfs(0, [], {0}, 0)
    
    if best_count == 0:
        best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
