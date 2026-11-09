"""Find maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import sys
import numpy as np

_LCM = 232792560  # LCM(1..22)

def _check_dissociated_np(vals):
    """Check if all 2^k subset sums of vals are distinct using NumPy."""
    k = len(vals)
    if k == 0:
        return True
    if k > 24: # Safety
        return False
    
    # Generate all subset sums
    # For k=22, 2^22 ~ 4 million integers.
    sums = np.array([0], dtype=np.int64)
    for v in vals:
        sums = np.concatenate((sums, sums + v))
        
    # Check distinctness by sorting and checking differences
    sums.sort()
    # np.diff(sums) > 0 ensures all adjacent elements are distinct
    # Since it's sorted, this implies all elements are distinct
    return np.all(np.diff(sums) > 0)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin (limit is 60s)
    
    # Precompute scaled values: w_i = LCM // i
    # Indices are 1..22
    scaled = [(_LCM // i) for i in range(1, 23)]
    
    best = []
    best_len = 0
    
    all_elements = list(range(1, 23))
    
    def greedy_init(order):
        current = []
        for idx in order:
            test_vals = [scaled[i-1] for i in current + [idx]]
            if _check_dissociated_np(test_vals):
                current.append(idx)
        return current

    # 1. Deterministic Greedy Initializations
    orders = [
        sorted(all_elements),          # 1, 2, ..., 22 (Large weights first)
        sorted(all_elements, reverse=True), # 22, ..., 1 (Small weights first)
        sorted(all_elements, key=lambda x: scaled[x-1]) # Smallest weights first
    ]
    
    for o in orders:
        if time.time() > deadline: break
        cand = greedy_init(o)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # 2. Randomized Local Search / Greedy Restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Random shuffle for greedy
        order = all_elements[:]
        random.shuffle(order)
        
        cand = greedy_init(order)
        
        # Local Improvement: Try to add any remaining element
        # This is crucial to maximize cardinality
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in all_elements:
                if idx in cand:
                    continue
                test_vals = [scaled[i-1] for i in cand + [idx]]
                if _check_dissociated_np(test_vals):
                    cand.append(idx)
                    improved = True
                    break # Restart loop as set changed
            
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
    return best
