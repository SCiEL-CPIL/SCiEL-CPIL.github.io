import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin
    
    # Precompute all t-subsets and k-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    all_k_subs = list(itertools.combinations(range(v), k))
    n_subs = len(all_t_subs)
    n_blocks = len(all_k_subs)
    
    # Create a mapping from subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute indices of t-subsets for each k-block
    num_subs_per_block = comb(k, t)
    block_sub_indices = np.zeros((n_blocks, num_subs_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_k_subs):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_sub_indices[i, j] = sub_to_idx[sub]
            
    # Greedy Cover using Vectorized Operations
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # uncovered: boolean array of shape (n_subs,)
        uncovered = np.ones(n_subs, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            # Compute gains for all blocks
            gains = uncovered[block_sub_indices].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            if max_gain == 0:
                idx = np.random.randint(n_blocks)
            else:
                candidates = np.where(gains == max_gain)[0]
                idx = np.random.choice(candidates)
            
            selected_blocks.append(all_k_subs[idx])
            
            # Update uncovered
            uncovered[block_sub_indices[idx]] = False
            
            if time.time() > deadline:
                break
        
        return selected_blocks

    # Local Search (Deletion)
    def local_search_remove(blocks):
        if not blocks:
            return blocks
            
        counts = np.zeros(n_subs, dtype=np.int32)
        
        for b in blocks:
            for sub in itertools.combinations(b, t):
                idx = sub_to_idx[sub]
                counts[idx] += 1
        
        redundant_mask = np.ones(len(blocks), dtype=bool)
        
        for i, b in enumerate(blocks):
            is_redundant = True
            for sub in itertools.combinations(b, t):
                idx = sub_to_idx[sub]
                if counts[idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                redundant_mask[i] = False
        
        if np.all(~redundant_mask):
            return blocks
            
        new_blocks = [b for i, b in enumerate(blocks) if not redundant_mask[i]]
        return new_blocks

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            prev_len = len(blocks)
            blocks = local_search_remove(blocks)
            if len(blocks) < prev_len:
                improved = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks.copy()
                    
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
            
        seed += 1
        
        if time.time() > deadline:
            break

    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return best_blocks
