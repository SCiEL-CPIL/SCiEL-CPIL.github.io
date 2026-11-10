# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Randomized deletion strategy with exact check for small sets.
    """
    n = 8
    deadline = time.time() + 590.0
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    active_list = list(range(N))
    
    def is_bad_5_tuple_indices(indices):
        p_idx = indices[0]
        q_indices = indices[1:5]
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        matrix = np.empty((4, 4), dtype=np.float64)
        for i, q_idx in enumerate(q_indices):
            q = grid[q_idx]
            q_sq = sq_norms[q_idx]
            matrix[i, 0:3] = q - p
            matrix[i, 3] = q_sq - p_sq
        det_val = np.linalg.det(matrix)
        return abs(det_val) < 0.5

    M_SAMPLES = 5000  # Increase samples to be safer
    
    while len(active_list) > 5 and time.time() < deadline - 10:
        found_violation = False
        
        # If the set is small, check all combinations
        if len(active_list) <= 50:
            # Check all 5-tuples
            for combo in itertools.combinations(active_list, 5):
                if is_bad_5_tuple_indices(combo):
                    # Remove the first point in the combo
                    remove_idx = combo[0]
                    active_list.remove(remove_idx)
                    found_violation = True
                    break
            if not found_violation:
                break # No violations in small set, done
            else:
                continue # Restart loop to check again after removal
        
        # For larger sets, use random sampling
        for _ in range(M_SAMPLES):
            if len(active_list) < 5:
                break
            sample = random.sample(active_list, 5)
            if is_bad_5_tuple_indices(sample):
                remove_idx = sample[0]
                active_list.remove(remove_idx)
                found_violation = True
                break
        
        if not found_violation:
            # No violation found in sampling
            break
            
    result = grid[active_list]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
