import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. Greedy Construction with multiple restarts
    best_solution = None
    best_size = float('inf')
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 35:
            break
            
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 25:
                break
                
            best_idx = -1
            best_gain = -1
            
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and best_idx != -1 and random.random() < 0.15:
                    if random.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        if len(selected_block_indices) < best_size:
            best_size = len(selected_block_indices)
            best_solution = selected_block_indices[:]
    
    if best_solution is None:
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        while uncovered_mask != 0:
            if time.time() > deadline - 15:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        best_solution = selected_block_indices
    
    # 4. Local Search: Redundancy Removal (iterative)
    current_blocks = best_solution[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    cover_count = [0] * n_t_subs
    for mask in current_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
        for i in range(len(current_blocks)):
            mask = current_masks[i]
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
                
            if is_critical:
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
    
    # 5. Swap-based local search: try replacing one block with another
    if time.time() < deadline - 10:
        # Build cover_count for current solution
        cover_count = [0] * n_t_subs
        for mask in current_masks:
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        # For each selected block, find its uniquely covered subsets
        # Try to swap with an unselected block that covers those subsets
        # and may enable further removals
        
        selected_set = set(current_blocks)
        
        # Multiple passes of swap improvement
        for swap_pass in range(3):
            if time.time() > deadline - 8:
                break
            
            improved_this_pass = False
            
            for i in range(len(current_blocks)):
                if time.time() > deadline - 8:
                    break
                    
                block_i = current_blocks[i]
                mask_i = current_masks[i]
                
                # Find subsets that only block_i covers
                unique_subsets = []
                m = mask_i
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique_subsets.append(idx)
                    m ^= lsb
                
                if not unique_subsets:
                    # Block can be removed (shouldn't happen after redundancy removal)
                    continue
                
                # Create a bitmask of unique subsets
                unique_mask = 0
                for idx in unique_subsets:
                    unique_mask |= (1 << idx)
                
                # Try to find an unselected block that covers all unique subsets
                # and has good overall coverage
                best_swap = -1
                best_swap_score = -1
                
                # Sample a subset of unselected blocks for efficiency
                unselected = [j for j in range(n_k_blocks) if j not in selected_set]
                
                # If too many, sample
                if len(unselected) > 5000:
                    random.shuffle(unselected)
                    unselected = unselected[:5000]
                
                for j in unselected:
                    mask_j = block_masks[j]
                    # Check if j covers all unique subsets of i
                    if (mask_j & unique_mask) == unique_mask:
                        # Score: how many new subsets does j cover?
                        new_coverage = (mask_j & ~mask_i).bit_count()
                        # Also consider how many subsets become redundant
                        # Simple score: new coverage minus cost
                        score = new_coverage
                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap = j
                
                if best_swap != -1:
                    # Perform the swap
                    # Remove block_i, add best_swap
                    # Update cover counts
                    m = mask_i
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    
                    m = block_masks[best_swap]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                    
                    current_blocks[i] = best_swap
                    current_masks[i] = block_masks[best_swap]
                    selected_set.remove(block_i)
                    selected_set.add(best_swap)
                    improved_this_pass = True
                    
                    # Run redundancy removal after swap
                    improved = True
                    while improved and time.time() < deadline - 8:
                        improved = False
                        new_current_blocks = []
                        new_current_masks = []
                        
                        for idx in range(len(current_blocks)):
                            mask = current_masks[idx]
                            is_critical = False
                            m = mask
                            while m:
                                lsb = m & -m
                                sub_idx = lsb.bit_length() - 1
                                if cover_count[sub_idx] == 1:
                                    is_critical = True
                                    break
                                m ^= lsb
                                
                            if is_critical:
                                new_current_blocks.append(current_blocks[idx])
                                new_current_masks.append(mask)
                            else:
                                m = mask
                                while m:
                                    lsb = m & -m
                                    sub_idx = lsb.bit_length() - 1
                                    cover_count[sub_idx] -= 1
                                    m ^= lsb
                                improved = True
                                
                        # Update selected_set
                        old_blocks = set(current_blocks)
                        current_blocks = new_current_blocks
                        current_masks = new_current_masks
                        new_set = set(current_blocks)
                        removed = old_blocks - new_set
                        added = new_set - old_blocks
                        selected_set -= removed
                        selected_set |= added
                        
                    # Re-index since blocks may have been removed
                    i = 0  # Restart from beginning after removals
            
            if not improved_this_pass:
                break
    
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
