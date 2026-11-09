import random
import numpy as np

def _greedy_solve_np(order):
    """
    Greedily constructs a B2[2] set from the given order of candidates using NumPy.
    Returns the list of chosen integers.
    """
    max_sum = 2000
    counts = np.zeros(max_sum + 1, dtype=np.int8)
    chosen = np.zeros(1000, dtype=np.int32)
    k = 0
    
    for x in order:
        # Check if adding x violates the B2[2] condition
        if k > 0:
            sums = x + chosen[:k]
            if np.any(counts[sums] >= 2):
                continue
        
        s_double = 2 * x
        if counts[s_double] >= 2:
            continue
        
        # Valid: add x
        chosen[k] = x
        k += 1
        
        # Update the count array
        if k > 1:
            update_sums = x + chosen[:k-1]
            counts[update_sums] += 1
            
        counts[s_double] += 1
        
    return chosen[:k].tolist()

def construct():
    # Base candidates
    candidates = list(range(1, 1001))
    
    best_set = []
    best_size = 0
    
    # 1. Try deterministic orders first (ascending, descending)
    deterministic_orders = [
        candidates,
        sorted(candidates, reverse=True)
    ]
    
    for order in deterministic_orders:
        current_set = _greedy_solve_np(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # 2. Try random restarts
    # NumPy allows for many more restarts
    num_restarts = 10000
    
    for _ in range(num_restarts):
        order = candidates[:]
        random.shuffle(order)
        current_set = _greedy_solve_np(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
