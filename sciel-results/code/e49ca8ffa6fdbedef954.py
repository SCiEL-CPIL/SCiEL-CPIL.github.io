import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    # 1. Greedy Heuristic for initial bound
    # Sort indices by weight descending
    idx_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in idx_sorted]
    
    best_sum = 0
    best_mask = 0
    curr_sum = 0
    curr_mask = 0
    for i in range(n):
        if curr_sum + sorted_weights[i] <= L - 1:
            curr_sum += sorted_weights[i]
            curr_mask |= (1 << i)
    best_sum = curr_sum
    best_mask = curr_mask

    # 2. Branch and Bound
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    time_exceeded = False
    
    def dfs(i, c_sum, c_mask):
        nonlocal best_sum, best_mask, time_exceeded
        
        if time_exceeded:
            return
            
        if time.time() > deadline:
            time_exceeded = True
            return
            
        if i == n:
            if c_sum > best_sum:
                best_sum = c_sum
                best_mask = c_mask
            return
            
        rem_cap = (L - 1) - c_sum
        if rem_cap < 0:
            return
            
        # Upper bound
        if c_sum + min(rem_cap, suffix_sums[i]) <= best_sum:
            return
            
        # Include
        if sorted_weights[i] <= rem_cap:
            dfs(i + 1, c_sum + sorted_weights[i], c_mask | (1 << i))
            
        # Exclude
        dfs(i + 1, c_sum, c_mask)

    dfs(0, 0, 0)
    
    # Map back to original indices
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(idx_sorted[i])
            
    return result_indices
