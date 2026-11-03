import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Generate all k-subsets and t-subsets
    points = np.arange(v)
    k_subsets = np.array(list(itertools.combinations(range(v), k)), dtype=np.int32)
    t_subsets = np.array(list(itertools.combinations(range(v), t)), dtype=np.int32)
    
    num_k = len(k_subsets)
    num_t = len(t_subsets)
    
    # Precompute point bitmasks
    point_masks = np.array([1 << i for i in range(v)], dtype=np.int32)
    
    # Precompute t_block_masks
    t_block_masks = np.bitwise_or.reduce(point_masks[t_subsets], axis=1) # (num_t,)
    
    # 2. Vectorized Mask Computation (Chunked)
    masks = np.zeros((num_k, num_t), dtype=np.uint8)
    chunk_size = 2000
    
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_subsets[start:end]
        k_block_masks = np.bitwise_or.reduce(point_masks[k_chunk], axis=1) # (chunk,)
        
        # Check subset condition: (k_mask & t_mask) == t_mask
        intersection = np.bitwise_and(k_block_masks[:, None], t_block_masks[None, :])
        masks[start:end, :] = (intersection == t_block_masks[None, :]).astype(np.uint8)
        
        if time.time() > deadline - 20:
            break

    # 3. Greedy Set Cover
    uncovered = np.ones(num_t, dtype=bool)
    selected_blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline - 5:
            break
            
        # Compute gains
        gains = masks @ uncovered
        
        # Find max gain
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Pick one randomly to allow diversity
        chosen_idx = candidates[np.random.randint(len(candidates))]
        
        selected_blocks.append(chosen_idx)
        uncovered[masks[chosen_idx] == 1] = False
        
    # 4. Reduction Phase (Lightweight)
    if time.time() < deadline - 10 and len(selected_blocks) > 0:
        sel_masks = masks[selected_blocks]
        counts = sel_masks.sum(axis=0)
        
        new_selected = []
        for b_idx in selected_blocks:
            block_mask = masks[b_idx]
            covered_idx = np.where(block_mask == 1)[0]
            if np.all(counts[covered_idx] > 1):
                # Redundant, remove it
                counts[block_mask == 1] -= 1
            else:
                new_selected.append(b_idx)
        
        selected_blocks = new_selected

    # 5. Return Result
    final_blocks = [list(k_subsets[i]) for i in selected_blocks]
    return final_blocks
