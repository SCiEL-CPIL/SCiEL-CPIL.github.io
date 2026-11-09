import time
import random
import itertools

_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def is_dissociated(subset_indices):
    """
    Checks if the subset of indices (0-based into _WEIGHTS) is dissociated.
    """
    if not subset_indices:
        return True
    
    # Map indices to weights
    # We can work with the indices directly or the weights.
    # Using weights is better for summing.
    weights = [_WEIGHTS[i] for i in subset_indices]
    k = len(weights)
    
    # Generate all subset sums
    # Optimization: if k is small, just iterate.
    # We use a set to store sums.
    sums = set()
    # Iterate over all 2^k masks
    # For speed, we can use a recursive or iterative DP approach, 
    # but for k <= 20, simple iteration is fine.
    
    # A faster way to generate subset sums:
    # Start with {0}
    # For each weight w, new_sums = {s + w for s in current_sums}
    # If any s + w is in current_sums or new_sums, it's not dissociated?
    # No, if s1 + w == s2, then s1 + w - s2 = 0. 
    # Actually, we just need to check if all 2^k sums are distinct.
    
    current_sums = {0}
    for w in weights:
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums or ns in new_sums:
                return False
            new_sums.add(ns)
        current_sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = 50.0  # 50s limit to be safe
    
    best_size = 0
    best_subset = []
    
    # 1. Try the seed
    seed_subset = [0, 1, 2, 4, 6, 10, 12, 16, 18] # Corresponds to 1,2,3,5,7,11,13,17,19
    # Wait, seed is [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # Indices: 0, 1, 2, 4, 6, 10, 12, 16, 18
    if is_dissociated(seed_subset):
        best_size = len(seed_subset)
        best_subset = seed_subset
    
    # 2. Random Greedy Restart
    # We want to maximize cardinality.
    # Strategy: Randomly shuffle the 22 items.
    # Iterate through them, add if valid.
    
    for _ in range(5000):  # Enough iterations
        if time.time() > deadline:
            break
            
        order = list(range(22))
        random.shuffle(order)
        
        current_subset = []
        for idx in order:
            # Try adding idx
            new_subset = current_subset + [idx]
            if is_dissociated(new_subset):
                current_subset = new_subset
        
        if len(current_subset) > best_size:
            best_size = len(current_subset)
            best_subset = current_subset
            
            # If we hit 22, we can stop? Unlikely.
            # If we hit a high number like 18, we can continue to see if 19 is possible.
            
    # 3. Local Search / Hill Climbing on the best found
    # Try to improve best_subset
    # We can try to add any missing element.
    # If we can't add, try to swap.
    
    # This part is optional if random greedy is good enough.
    # Given the "inspiration" that multiple greedy orderings work, and the observed score, 
    # random greedy is likely sufficient to find 16+.
    
    # Convert best_subset (indices) to 1-based values
    # best_subset contains indices 0..21 corresponding to values 1..22
    return [i + 1 for i in best_subset]
