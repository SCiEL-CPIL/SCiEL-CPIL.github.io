"""Covering C(19,8,5): randomized minimal cover via iterative peeling to fixed point."""
import numpy as np
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    block_t_list = []
    for block in k_blocks:
        covered = np.array([t_index[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_t_list.append(covered)
    
    # Precompute full coverage
    full_coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        full_coverage[covered] += 1
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple trials to find a better minimal cover
    for trial in range(20):
        random.seed(42 + trial)
        order = list(range(n_k))
        random.shuffle(order)
        
        active = np.ones(n_k, dtype=bool)
        coverage = full_coverage.copy()
        
        # Iterate until no more blocks can be removed (fixed point)
        removed_any = True
        while removed_any:
            removed_any = False
            # We need to check blocks in the shuffled order.
            # To be efficient, we can just iterate through the shuffled list.
            # If a block is removed, we don't need to check it again in this pass?
            # Actually, if we remove a block, it might make others non-essential.
            # So we must check all active blocks.
            
            for i in order:
                if not active[i]:
                    continue
                
                # Check if block i is essential
                if np.any(coverage[block_t_list[i]] == 1):
                    continue
                
                # Remove block i
                active[i] = False
                coverage[block_t_list[i]] -= 1
                removed_any = True
                
        count = np.sum(active)
        if count < best_count:
            best_count = count
            best_blocks = [list(k_blocks[i]) for i in range(n_k) if active[i]]
            
    return best_blocks
