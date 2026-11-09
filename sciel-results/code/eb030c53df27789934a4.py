# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 50.0  # 50 seconds safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    L = reduce(lcm, denominators)
    
    # Convert to integer values
    vals = [L // d for d in denominators]
    
    # Suffix sums
    suf = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + vals[i]
    
    best_deficit = L  # Max possible deficit
    best_indices = []
    
    # To track current path
    current_path = []
    
    def dfs(idx, current_sum):
        nonlocal best_deficit, best_indices
        nonlocal deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: if the maximum possible sum from this state 
        # does not yield a deficit smaller than current best, prune.
        max_possible_sum = current_sum + suf[idx]
        if L - max_possible_sum >= best_deficit:
            return
            
        # Check if current sum is a valid candidate
        # It is always < L if we reached here (except we check before adding)
        # But we might have started with 0.
        # We need non-empty.
        if current_path:
            deficit = L - current_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_indices = list(current_path)
        
        if idx >= n:
            return
            
        # If taking all remaining keeps us below L, 
        # then the best sum in this branch is current_sum + suf[idx]
        if current_sum + suf[idx] < L:
            total_sum = current_sum + suf[idx]
            deficit = L - total_sum
            if deficit < best_deficit:
                best_deficit = deficit
                # Indices are current_path + list(range(idx, n))
                best_indices = list(current_path) + list(range(idx, n))
            return
            
        # Option 1: Include idx
        # Check if adding val[idx] keeps sum < L
        if current_sum + vals[idx] < L:
            current_path.append(idx)
            dfs(idx + 1, current_sum + vals[idx])
            current_path.pop()
            
        # Option 2: Exclude idx
        dfs(idx + 1, current_sum)
        
    dfs(0, 0)
    
    return best_indices
# EVOLVE-BLOCK-END
