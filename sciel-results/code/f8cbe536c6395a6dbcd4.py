import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal fractions
    fracs = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum(fracs[i] + ... + fracs[n-1])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + fracs[i]
    
    # 1. Get a strong initial solution using Greedy
    # Greedy: Iterate from smallest denominator (largest reciprocal) to largest.
    # If adding it keeps sum < 1, add it.
    greedy_sum = Fraction(0)
    greedy_indices = []
    for i in range(n):
        if greedy_sum + fracs[i] < 1:
            greedy_sum += fracs[i]
            greedy_indices.append(i)
            
    # Ensure we have a valid non-empty solution
    if not greedy_indices:
        # Fallback: just 1/2
        greedy_indices = [0]
        greedy_sum = fracs[0]
        
    best_sum = greedy_sum
    best_indices = greedy_indices[:]
    
    # 2. Branch and Bound Search
    # We use a stack for DFS.
    # State: (index, current_sum, current_indices_list)
    stack = [(0, Fraction(0), [])]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, curr_sum, curr_indices = stack.pop()
        
        if idx == n:
            if curr_sum < 1 and curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = curr_indices[:]
            continue
        
        # Pruning 1: Even if we add all remaining items, we can't beat best_sum
        if curr_sum + suffix_sum[idx] <= best_sum:
            continue
            
        # Pruning 2: Current sum is already >= 1 (invalid)
        if curr_sum >= 1:
            continue
        
        # Option 1: Exclude current denominator
        # Push exclude branch
        stack.append((idx + 1, curr_sum, curr_indices))
        
        # Option 2: Include current denominator
        new_sum = curr_sum + fracs[idx]
        if new_sum < 1:
            # Push include branch
            # Using list concatenation creates a new list
            stack.append((idx + 1, new_sum, curr_indices + [idx]))
        
    return best_indices
