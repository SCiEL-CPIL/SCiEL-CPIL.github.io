import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets and the indices of t-subsets they contain
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map each k-subset to a list of t-subset indices
    # Shape: (n_k, C(k, t))
    # C(k, t) is fixed for a given v, k, t
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    # We need to map the local indices (0..k-1) to global indices (0..v-1)
    # Actually, itertools.combinations(range(v), k) gives the actual points.
    # So for a block B (tuple of points), we generate combinations of points from B.
    
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subsets):
        for j, t_combo in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[t_combo]
            
    # 3. Greedy Set Cover
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    
    while np.any(uncovered):
        if time.time() - start_time > deadline - 5.0:
            break
            
        # Calculate gain for each block
        # uncovered[indices] -> shape (n_k, n_t_in_k)
        # sum along axis 1 -> shape (n_k,)
        gains = uncovered[block_t_indices].sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain
        # To avoid bias, we can pick one randomly or the first one.
        # Picking the first one is deterministic.
        best_idx = np.argmax(gains)
        
        selected_blocks.append(k_subsets[best_idx])
        
        # Update uncovered
        uncovered[block_t_indices[best_idx]] = False
        
    # 4. Post-processing: Remove redundant blocks
    # Build coverage count
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for block in selected_blocks:
        for t_combo in itertools.combinations(block, t):
            coverage_count[t_sub_to_idx[t_combo]] += 1
            
    # Try to remove blocks
    # We need to check if a block is redundant.
    # A block is redundant if all its t-subsets are covered by at least 2 blocks.
    # i.e., min(coverage_count[s] for s in block) > 1
    
    changed = True
    while changed:
        if time.time() - start_time > deadline - 2.0:
            break
        changed = False
        for i in range(len(selected_blocks)):
            block = selected_blocks[i]
            # Check if redundant
            is_redundant = True
            for t_combo in itertools.combinations(block, t):
                if coverage_count[t_sub_to_idx[t_combo]] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                # Update counts
                for t_combo in itertools.combinations(block, t):
                    coverage_count[t_sub_to_idx[t_combo]] -= 1
                selected_blocks.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return [list(b) for b in selected_blocks]
