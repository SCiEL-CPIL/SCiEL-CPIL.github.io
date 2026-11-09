import random
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 55.0  # Safety margin for 60s limit
    N = 4096
    MAX_SIZE = 80
    MAX_SUM = 3 * N
    
    # Precompute candidates
    candidates = list(range(1, N + 1))
    
    def build_greedy(order):
        selected = []
        # Use a boolean array for O(1) vectorized lookup of sums
        # Size 3*N + 1 is sufficient (max sum is 3*4096)
        sum_exists = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in order:
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 2:
                break
            
            # Generate new sums involving x
            new_sums = []
            
            # 1. 3x
            new_sums.append(3 * x)
            
            # 2. 2x + y for y in selected
            # 3. x + y + z for y, z in selected
            if selected:
                sel_arr = np.array(selected)
                
                # 2x + y
                new_sums.extend((2 * x + sel_arr).tolist())
                
                # x + y + z
                # Create meshgrid of selected elements
                y, z = np.meshgrid(sel_arr, sel_arr)
                pair_sums = y + z
                new_sums.extend((x + pair_sums).flatten().tolist())
            
            new_sums_arr = np.array(new_sums)
            
            # Check for collisions
            # If any new sum is already marked as existing, x is invalid.
            if np.any(sum_exists[new_sums_arr]):
                continue
            
            # If valid, mark all new sums as existing
            sum_exists[new_sums_arr] = True
            selected.append(x)
        
        return selected

    best_set = []
    
    # 1. Try natural order (ascending)
    current_set = build_greedy(candidates)
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # 2. Try descending order
    current_set = build_greedy(sorted(candidates, reverse=True))
    if len(current_set) > len(best_set):
        best_set = current_set
        
    # 3. Try random permutations
    attempts = 0
    while time.time() < DEADLINE - 2.0:
        random.shuffle(candidates)
        current_set = build_greedy(candidates)
        if len(current_set) > len(best_set):
            best_set = current_set
        attempts += 1
        
        # If we found the maximum possible size, we can stop early
        if len(best_set) == MAX_SIZE:
            break
            
    return best_set
