import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # 3. Build dense coverage matrix M (num_blocks x num_t)
    # M[i, j] = 1 if block i covers t-subset j
    # Using uint8 to save memory
    M = np.zeros((num_blocks, num_t), dtype=np.uint8)
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            M[i, idx] = 1

    # 4. Greedy Set Cover
    uncovered = np.ones(num_t, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 15.0:
            break
            
        # Calculate gain for each block: number of uncovered t-subsets it covers
        gains = M @ uncovered
        
        # Find block with max gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        
        # Update uncovered: set to 0 where the chosen block covers
        M_row = M[best_idx]
        uncovered[M_row == 1] = 0

    # 5. Redundancy Removal
    if not selected_indices:
        return []
        
    selected_indices = np.array(selected_indices, dtype=np.int64)
    
    # Compute coverage counts
    M_selected = M[selected_indices]
    coverage_counts = M_selected.sum(axis=0)
    
    keep_mask = np.ones(len(selected_indices), dtype=bool)
    
    for i in range(len(selected_indices)):
        if not keep_mask[i]:
            continue
            
        if time.time() > deadline - 10.0:
            break
            
        block_row = M[selected_indices[i]]
        counts_of_covered = coverage_counts[block_row == 1]
        
        if len(counts_of_covered) > 0 and np.min(counts_of_covered) == 1:
            # Essential
            continue
        else:
            # Redundant
            keep_mask[i] = False
            coverage_counts -= block_row
            
    final_indices = selected_indices[keep_mask]
    n_final = len(final_indices)
    
    # 6. Local Search Refinement: try to replace pairs of blocks with a single block
    # This can reduce the total count by 1 per successful swap
    if time.time() < deadline - 8.0 and n_final > 0:
        selected_set = set(final_indices.tolist())
        M_sel = M[list(selected_set)]
        cov_counts = M_sel.sum(axis=0).astype(np.int32)
        
        # Precompute which t-subsets each selected block uniquely covers
        unique_cover = []
        for i in range(n_final):
            row = M_sel[i]
            unique_mask = (cov_counts == 1) & (row == 1)
            unique_cover.append(np.where(unique_mask)[0])
        
        # Try to find pairs (i, j) where block i can be removed if we add a new block
        # that covers all unique t-subsets of block i
        removed_any = True
        while removed_any and time.time() < deadline - 5.0:
            removed_any = False
            n_cur = len(selected_set)
            if n_cur == 0:
                break
            
            sel_list = list(selected_set)
            M_cur = M[sel_list]
            cov_cur = M_cur.sum(axis=0).astype(np.int32)
            
            # Find blocks that are "almost redundant" - their unique coverage is small
            for i in range(n_cur):
                if time.time() > deadline - 5.0:
                    break
                
                row_i = M_cur[i]
                unique_i = (cov_cur == 1) & (row_i == 1)
                unique_count = unique_i.sum()
                
                # Only consider blocks with small unique coverage (hard to remove)
                # Actually, we want to REMOVE a block, so we need to cover its unique parts
                # Try to find a single block that covers all unique t-subsets of block i
                if unique_count == 0:
                    # Block i is fully redundant
                    selected_set.discard(sel_list[i])
                    removed_any = True
                    break
                
                if unique_count <= 10:  # Only try if unique coverage is small
                    # Find all blocks that cover all unique t-subsets of block i
                    unique_indices = np.where(unique_i)[0]
                    # For each candidate block, check if it covers all unique t-subsets
                    # This is expensive, so only do it for a subset of candidates
                    candidates = np.where(M[:, unique_indices].all(axis=1))[0]
                    candidates = candidates.tolist()
                    
                    if len(candidates) > 0:
                        # Pick a random candidate (or the one that covers most other things)
                        random.shuffle(candidates)
                        new_block = candidates[0]
                        
                        # Check if adding new_block and removing i maintains coverage
                        # The new block must cover all unique t-subsets of i
                        # And after removing i, no t-subset should become uncovered
                        new_row = M[new_block]
                        # Simulate: remove i, add new_block
                        # t-subsets that were only covered by i must be covered by new_block
                        # t-subsets that were covered by i and others are still covered
                        # So we just need: for all j where cov_cur[j] == 1 and row_i[j] == 1, new_row[j] == 1
                        if np.all(new_row[unique_i]):
                            selected_set.discard(sel_list[i])
                            if new_block not in selected_set:
                                selected_set.add(new_block)
                            removed_any = True
                            break
        
        final_indices = np.array(sorted(selected_set), dtype=np.int64)
    
    final_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return final_blocks
