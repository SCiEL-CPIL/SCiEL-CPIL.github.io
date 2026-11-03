import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Leave 5 seconds for safety
    deadline = start_time + 115.0 

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(k_subsets)
    
    # Precompute indices of t-subsets for each k-subset
    # Shape: (n_k_subsets, C(k, t))
    n_t_in_k = len(list(itertools.combinations(range(k), t)))
    block_indices = np.zeros((n_k_subsets, n_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subsets):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_indices[i, j] = t_subset_to_idx[sub]
            
    # 2. Helper: Greedy Run
    def greedy_run(seed):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            # This creates a temporary array of size (n_k_subsets, n_t_in_k)
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            
            if max_gain == 0:
                # Should not happen if uncovered.any() is true and blocks exist
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one
            if len(candidates) == 1:
                pick = candidates[0]
            else:
                # Random tie-breaking
                pick = candidates[rng.integers(len(candidates))]
                
            blocks.append(pick)
            
            # Update uncovered
            uncovered[block_indices[pick]] = False
            
            # Time check
            if time.time() > deadline:
                break
                
        # Convert indices to actual blocks
        actual_blocks = [list(k_subsets[b]) for b in blocks]
        return actual_blocks

    # 3. Helper: Remove Redundant Blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Count coverage for each t-subset
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        
        # Initialize counts
        for b in blocks:
            cover_count[block_indices[b]] += 1
            
        new_blocks = []
        for b in blocks:
            # Check if all t-subsets in this block are covered by other blocks
            # i.e., cover_count[subset] > 1
            if np.min(cover_count[block_indices[b]]) > 1:
                # Safe to remove
                cover_count[block_indices[b]] -= 1
            else:
                # Must keep
                new_blocks.append(b)
                
        return [list(k_subsets[b]) for b in new_blocks]

    # 4. Main Loop
    best_blocks = []
    best_len = float('inf')
    
    # Run greedy with seed 0
    blocks = greedy_run(0)
    blocks = remove_redundant(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # Run more restarts if time allows
    seed = 1
    while time.time() < deadline - 10.0: # Keep 10s buffer for final processing
        blocks = greedy_run(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
        seed += 1
        
    return best_blocks
