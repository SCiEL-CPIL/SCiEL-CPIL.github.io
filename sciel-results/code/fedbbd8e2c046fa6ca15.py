import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Precompute all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Build coverage matrix
    # Using float32 for fast dot product
    coverage_matrix = np.zeros((num_blocks, num_t), dtype=np.float32)
    
    for i, block in enumerate(k_subsets):
        for ts in itertools.combinations(block, t):
            j = t_to_id[ts]
            coverage_matrix[i, j] = 1.0
            
    def greedy_solve(rng):
        uncovered = np.ones(num_t, dtype=np.float32)
        chosen = []
        
        while True:
            # Check if all covered
            # Using sum is faster than any() for float arrays sometimes?
            # Or just track remaining count?
            # Let's use sum for simplicity, it's O(N)
            if uncovered.sum() == 0:
                break
            
            # Calculate gain: number of uncovered subsets covered by each block
            # Matrix-vector multiplication
            gains = coverage_matrix @ uncovered
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            # Due to floating point, exact match might be tricky?
            # But values are integers 0..7. So exact match is fine.
            candidates = np.where(gains == max_gain)[0]
            
            if rng:
                idx = rng.choice(candidates)
            else:
                idx = candidates[0]
                
            chosen.append(idx)
            
            # Update uncovered: subtract the coverage of the chosen block
            # Only where uncovered is 1
            # uncovered = np.where(uncovered > 0, uncovered - coverage_matrix[idx], 0)
            # Or simply:
            # uncovered[coverage_matrix[idx] > 0] = 0
            # But we need to ensure we don't go negative if we use subtraction.
            # Since we set to 0, it's safe.
            uncovered[coverage_matrix[idx] > 0] = 0
            
        return chosen

    def delete_redundant(chosen_indices):
        if not chosen_indices:
            return []
            
        # Count coverage using integer array for precision
        # We can reuse the float matrix but cast?
        # Better to keep an integer version or just use the float one for logic?
        # Logic: counts[j] > 1
        # If we use float matrix, counts will be float.
        counts = np.zeros(num_t, dtype=np.float32)
        for idx in chosen_indices:
            counts += coverage_matrix[idx]
            
        valid = np.ones(len(chosen_indices), dtype=bool)
        changed = True
        
        while changed:
            changed = False
            for i in range(len(chosen_indices)):
                if not valid[i]:
                    continue
                    
                idx = chosen_indices[i]
                mask = coverage_matrix[idx] > 0
                
                # Check if all covered subsets have count > 1
                if np.all(counts[mask] > 1):
                    valid[i] = False
                    counts -= coverage_matrix[idx]
                    changed = True
                    
        return [chosen_indices[i] for i in range(len(chosen_indices)) if valid[i]]

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial deterministic greedy
    best_indices = greedy_solve(None)
    best_indices = delete_redundant(best_indices)
    best_size = len(best_indices)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        current_indices = greedy_solve(rng)
        current_indices = delete_redundant(current_indices)
        
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_indices = current_indices
            
    result_blocks = [list(k_subsets[i]) for i in best_indices]
    
    return result_blocks
