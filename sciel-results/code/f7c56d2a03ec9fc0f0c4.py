import math
import numpy as np
import random

def construct():
    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        ratio = n_diffs / n_sums
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    best_U = [0]
    best_score = -1
    
    # Try to optimize sets of different sizes
    for N in [4, 5, 6, 7]:
        # Initialize with a random set
        # Ensure 0 is in the set
        U = [0] + random.sample(range(1, 100), N-1)
        U = sorted(U)
        
        current_score = score(U)
        
        # Simple Hill Climbing
        for _ in range(1000):
            # Pick a random element to change
            i = random.randint(0, N-1)
            if i == 0:
                continue # Keep 0 fixed
            
            old_val = U[i]
            # Try a new random value
            new_val = random.randint(1, 150)
            
            U[i] = new_val
            U = sorted(U)
            
            new_score = score(U)
            
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                U[i] = old_val
                U = sorted(U)
        
        if current_score > best_score:
            best_score = current_score
            best_U = U
            
    return best_U
