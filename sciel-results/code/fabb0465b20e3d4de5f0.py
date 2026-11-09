import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    max_val = 4096
    max_sum = 3 * max_val
    sum_size = max_sum + 1
    
    best_set = []
    best_size = 0
    
    # Pre-allocate candidates
    candidates = np.arange(1, max_val + 1, dtype=np.int32)
    
    def try_greedy(order):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        current_set = []
        current_arr = np.array([], dtype=np.int32)
        
        # Use a bytearray for fast O(1) lookup of used sums
        used_sums = bytearray(sum_size)
        
        for x in order:
            if time.time() > deadline:
                break
            
            x_int = int(x)
            
            # Generate new sums using NumPy
            # 1. 3x
            sums_1 = np.array([3 * x_int], dtype=np.int32)
            
            if len(current_arr) > 0:
                # 2. 2x + y
                sums_2 = 2 * x_int + current_arr
                
                # 3. x + 2y
                sums_3 = x_int + 2 * current_arr
                
                # 4. x + y + z (i < j)
                # Create matrix of all pairs, take upper triangle
                # current_arr is shape (k,)
                # We want i < j
                # We can use indices
                idx_i, idx_j = np.triu_indices(len(current_arr), k=1)
                sums_4 = x_int + current_arr[idx_i] + current_arr[idx_j]
                
                # Combine all new sums
                new_sums = np.concatenate([sums_1, sums_2, sums_3, sums_4])
            else:
                new_sums = sums_1
            
            # Check internal collisions
            # Unique count must match total count
            if len(np.unique(new_sums)) != len(new_sums):
                continue
                
            # Check collision with existing sums
            # used_sums is a bytearray, so we can index it with an array of ints
            # If any sum in new_sums has used_sums[sum] == 1, it's a collision
            if np.any(used_sums[new_sums]):
                continue
                
            # Valid
            # Update used_sums
            used_sums[new_sums] = 1
            current_set.append(x_int)
            current_arr = np.append(current_arr, x_int)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 40: # Theoretical max
                return

    # 1. Deterministic Ascending
    try_greedy(candidates)
    
    # 2. Deterministic Descending
    try_greedy(candidates[::-1])
    
    # 3. Random Trials
    # Use a fixed seed for reproducibility if needed, but random is fine
    # We want diverse orderings.
    
    while time.time() < deadline:
        # Shuffle a copy to avoid modifying the original if we wanted to reuse it, 
        # but np.random.permutation is fast
        order = np.random.permutation(candidates)
        try_greedy(order)
        
    return best_set
