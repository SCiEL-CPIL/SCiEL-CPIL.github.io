import itertools
import time
import sys
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy heuristic followed by 
    iterative redundancy pruning and 2-for-1 block replacement local search.
    """
    deadline = time.monotonic() + 115.0  # Safety margin under 120s
    points = list(range(v))
    
    # 1. Precomputation
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute block masks and t-subset indices
    block_masks = [0] * num_kblocks
    block_tsub_indices = [None] * num_kblocks
    
    for i, b in enumerate(all_kblocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    full_mask = (1 << num_tsubs) - 1
    
    # Helper to check if a mask covers a set of indices (represented as mask)
    def covers(mask_a, mask_b):
        return (mask_a & mask_b) == mask_b

    # 2. Greedy Construction
    def greedy_init():
        selected_blocks = []
        uncovered_mask = full_mask
        while uncovered_mask:
            if time.monotonic() > deadline:
                break
            best_gain = 0
            best_block_idx = -1
            for i in range(num_kblocks):
                common = block_masks[i] & uncovered_mask
                if common:
                    gain = common.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = i
            if best_gain == 0:
                break
            selected_blocks.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        return selected_blocks

    selected_blocks = greedy_init()
    
    # Helper to update counts and check redundancy
    def update_counts(selected, counts):
        # Reset counts
        for t_idx in range(num_tsubs):
            counts[t_idx] = 0
        for b_idx in selected:
            for t_idx in block_tsub_indices[b_idx]:
                counts[t_idx] += 1

    counts = [0] * num_tsubs
    update_counts(selected_blocks, counts)
    
    # 3. Local Search
    def local_search(selected):
        nonlocal counts
        update_counts(selected, counts)
        
        while time.monotonic() < deadline:
            improved = False
            
            # Phase 1: Redundancy Removal
            # Remove blocks that have no unique t-subsets
            new_selected = []
            for b_idx in selected:
                redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Decrement counts
                    for t_idx in block_tsub_indices[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            
            if improved:
                selected = new_selected
                continue # Restart loop as counts changed
            
            # Phase 2: 2-for-0 or 2-for-1 Replacement
            n_sel = len(selected)
            found_replacement = False
            for i in range(n_sel):
                if time.monotonic() > deadline:
                    return selected
                for j in range(i + 1, n_sel):
                    # Temporarily remove i and j
                    # Calculate U: t-subsets that become uncovered
                    U_mask = 0
                    for t_idx in block_tsub_indices[selected[i]]:
                        if counts[t_idx] == 1:
                            U_mask |= (1 << t_idx)
                    for t_idx in block_tsub_indices[selected[j]]:
                        if counts[t_idx] == 1:
                            U_mask |= (1 << t_idx)
                    
                    if U_mask == 0:
                        # 2-for-0: Both are redundant in combination
                        # Remove both
                        for t_idx in block_tsub_indices[selected[i]]:
                            counts[t_idx] -= 1
                        for t_idx in block_tsub_indices[selected[j]]:
                            counts[t_idx] -= 1
                        new_sel = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                        selected = new_sel
                        found_replacement = True
                        break
                    else:
                        # 2-for-1: Find a block k that covers U
                        # Pick a t from U to find candidates
                        # Get the first set bit in U_mask
                        t_0 = (U_mask & -U_mask).bit_length() - 1
                        # Find all blocks covering t_0
                        # We can precompute tsub_to_blocks
                        # But we don't have it. We can iterate all blocks? No, too slow.
                        # Let's precompute tsub_to_blocks outside? 
                        # For now, iterate all kblocks and check if they cover t_0? 
                        # That is O(num_kblocks). 
                        # Better: Precompute a list of blocks for each t-subset.
                        
                        # Let's assume we have tsub_to_blocks precomputed.
                        # I'll add it in the main scope.
                        
                        pass # Placeholder, will implement below
            if found_replacement:
                continue # Restart loop
            
            # If no improvement, break
            break
            
        return selected

    # Precompute tsub_to_blocks for faster candidate search
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for t_idx in block_tsub_indices[i]:
            tsub_to_blocks[t_idx].append(i)

    # Re-run local search with the precomputed map
    # We need to modify the local_search function to use tsub_to_blocks
    # Let's rewrite the local search logic more cleanly.

    def improved_local_search(selected):
        nonlocal counts
        update_counts(selected, counts)
        
        while time.monotonic() < deadline:
            improved = False
            
            # Phase 1: Redundancy Removal
            new_selected = []
            for b_idx in selected:
                redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_tsub_indices[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            
            if improved:
                selected = new_selected
                continue
            
            # Phase 2: 2-for-0 or 2-for-1
            n_sel = len(selected)
            found = False
            for i in range(n_sel):
                if time.monotonic() > deadline:
                    break
                for j in range(i + 1, n_sel):
                    # Calculate U
                    U_mask = 0
                    for t_idx in block_tsub_indices[selected[i]]:
                        if counts[t_idx] == 1:
                            U_mask |= (1 << t_idx)
                    for t_idx in block_tsub_indices[selected[j]]:
                        if counts[t_idx] == 1:
                            U_mask |= (1 << t_idx)
                    
                    if U_mask == 0:
                        # Remove both
                        for t_idx in block_tsub_indices[selected[i]]:
                            counts[t_idx] -= 1
                        for t_idx in block_tsub_indices[selected[j]]:
                            counts[t_idx] -= 1
                        selected = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                        found = True
                        break
                    else:
                        # Find candidate k
                        t_0 = (U_mask & -U_mask).bit_length() - 1
                        candidates = tsub_to_blocks[t_0]
                        for k_idx in candidates:
                            if k_idx == selected[i] or k_idx == selected[j]:
                                continue
                            # Check if k covers U
                            if (block_masks[k_idx] & U_mask) == U_mask:
                                # Replace i, j with k
                                for t_idx in block_tsub_indices[selected[i]]:
                                    counts[t_idx] -= 1
                                for t_idx in block_tsub_indices[selected[j]]:
                                    counts[t_idx] -= 1
                                for t_idx in block_tsub_indices[k_idx]:
                                    counts[t_idx] += 1
                                # Remove i, j, add k
                                new_sel = [b for idx, b in enumerate(selected) if idx != i and idx != j]
                                new_sel.append(k_idx)
                                selected = new_sel
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
            
            if found:
                continue
            
            # If stuck, try a random swap to escape local minimum
            if len(selected) > 2:
                # Pick a random block to swap
                idx_to_swap = random.randint(0, len(selected) - 1)
                old_block = selected[idx_to_swap]
                # Pick a random new block
                new_block = random.randint(0, num_kblocks - 1)
                if new_block != old_block and new_block not in selected:
                    # Check if this swap is "okay" (doesn't uncover too much)
                    # Count uncovered t-subsets after swap
                    # This is expensive. 
                    # Simpler: Just do it, then run redundancy/pruning.
                    # If it makes things worse, we might get stuck again.
                    # Let's only do this if we have time.
                    if time.monotonic() < deadline - 5:
                         # Perform swap
                         for t_idx in block_tsub_indices[old_block]:
                             counts[t_idx] -= 1
                         for t_idx in block_tsub_indices[new_block]:
                             counts[t_idx] += 1
                         selected[idx_to_swap] = new_block
                         # Do NOT set improved=True to allow immediate retry of pruning
                         # But we should break the inner loop to restart the outer loop
                         break
            
            # If no improvement and no swap, break
            break
            
        return selected

    selected_blocks = improved_local_search(selected_blocks)
    
    # Final Output
    final_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return final_blocks
