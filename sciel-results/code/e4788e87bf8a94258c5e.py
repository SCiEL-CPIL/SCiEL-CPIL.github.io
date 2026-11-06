import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = reduce(lcm, denominators)
    
    # Create list of (value, original_index)
    # Sort by value descending to find good solutions early
    items = [(L // d, i) for i, d in enumerate(denominators)]
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_sums[i] is sum of vals from i to end
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    best_sum = 0
    best_mask = 0
    
    def dfs(idx, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        if time.time() - start_time > timeout:
            return
            
        if current_sum >= L:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = current_mask
            return
            
        # Pruning: even if we take all remaining, we can't beat best_sum
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        val, orig_idx = items[idx]
        
        # Option 1: Include current item
        next_sum = current_sum + val
        if next_sum < L:
            dfs(idx + 1, next_sum, current_mask | (1 << orig_idx))
            
        # Option 2: Exclude current item
        dfs(idx + 1, current_sum, current_mask)
        
    dfs(0, 0, 0)
    
    # Reconstruct indices from mask
    selected = []
    for i in range(n):
        if best_mask & (1 << i):
            selected.append(i)
            
    return selected
