import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Calculate LCM of 2..48
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Weights are L // d
    # We want sum(weights) < L
    # Maximize sum(weights)
    
    # Create list of (weight, index)
    items = [(L // d, i) for i, d in enumerate(denominators)]
    
    # Sort by weight descending (largest contribution first)
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums of weights
    # suffix[i] = sum of weights from index i to end
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + items[i][0]
        
    # Greedy initialization to get a strong initial bound
    best_sum = 0
    best_indices = []
    current_sum = 0
    for i in range(n):
        w, idx = items[i]
        if current_sum + w < L:
            current_sum += w
            best_indices.append(idx)
    best_sum = current_sum
    
    # DFS with Branch and Bound
    # Use a mutable list for indices to avoid copying
    curr_indices = []
    
    def dfs(idx, curr_sum):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning 1: Cannot beat best
        if curr_sum + suffix[idx] <= best_sum:
            return
            
        if idx == n:
            if curr_sum < L and curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = curr_indices[:]
            return
        
        # Option 1: Include item idx
        w, orig_idx = items[idx]
        new_sum = curr_sum + w
        if new_sum < L:
            curr_indices.append(orig_idx)
            dfs(idx + 1, new_sum)
            curr_indices.pop()
            
        # Option 2: Exclude item idx
        dfs(idx + 1, curr_sum)
        
    # Start DFS
    dfs(0, 0)
    
    # Verify result
    # best_indices are original indices
    # Check if valid
    from fractions import Fraction
    total_frac = sum(Fraction(1, denominators[i]) for i in best_indices)
    if total_frac < 1:
        return best_indices
    else:
        # Fallback to known good solution if something went wrong
        return [0, 1, 5, 41]

if __name__ == "__main__":
    construct()
