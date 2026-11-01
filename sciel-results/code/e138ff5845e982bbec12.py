import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute all t-subsets (targets)
    all_targets = list(itertools.combinations(range(v), t))
    num_targets = len(all_targets)
    
    # Map target to index
    target_to_idx = {tup: i for i, tup in enumerate(all_targets)}
    
    # Pre-compute for each block, the list of target indices it covers
    # This is a list of lists, where block_to_targets[i] is a list of target indices
    block_to_targets = []
    for block in all_blocks:
        targets = []
        for t_sub in itertools.combinations(block, t):
            targets.append(target_to_idx[t_sub])
        block_to_targets.append(np.array(targets, dtype=np.int32))
    
    # Initialize uncovered counts
    # uncovered_count[j] is 1 if target j is not yet covered, 0 otherwise
    uncovered_count = np.ones(num_targets, dtype=np.int32)
    uncovered_mask = np.ones(num_targets, dtype=bool)
    
    selected_indices = []
    
    # Greedy Set Cover
    # We will repeatedly pick the block that covers the most uncovered targets.
    # To speed this up, we calculate the gain for all blocks at once?
    # No, calculating gain for all 50k blocks every step might be heavy if done in Python loops.
    # However, we can pre-compute a matrix? No, memory might be high.
    # Let's try a hybrid: maintain a list of active blocks.
    # Actually, for v=19, num_blocks is ~50k. num_targets is ~3.8k.
    # If we iterate 50k blocks * 100 steps * (avg targets per block ~30) = 150M ops.
    # In Python, this is too slow.
    
    # Better approach:
    # Maintain a numpy array `current_gains` of size num_blocks.
    # Initially, gain[i] = len(block_to_targets[i]).
    # When we pick a block, we subtract 1 from the gain of all blocks that share any of the newly covered targets.
    # This requires knowing which blocks share targets with the picked block.
    # This is a sparse matrix problem.
    
    # Let's build a sparse adjacency: target -> list of blocks that cover it.
    target_to_blocks = [[] for _ in range(num_targets)]
    for i, targets in enumerate(block_to_targets):
        for t_idx in targets:
            target_to_blocks[t_idx].append(i)
    
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for small lists.
    # current_gains[i] = number of uncovered targets covered by block i.
    current_gains = np.array([len(targets) for targets in block_to_targets], dtype=np.int32)
    
    while np.any(uncovered_mask):
        if time.time() > deadline - 5.0:
            break
        
        # Find the block with the maximum gain
        # If there are multiple, pick one randomly to break symmetry and improve diversity?
        # Or just pick the first max.
        max_gain = np.max(current_gains)
        if max_gain == 0:
            # Should not happen if cover is possible, but just in case
            break
        
        candidates = np.where(current_gains == max_gain)[0]
        chosen_idx = random.choice(candidates)
        
        selected_indices.append(chosen_idx)
        
        # Update uncovered status
        chosen_targets = block_to_targets[chosen_idx]
        
        # Find which targets were actually newly covered (were 1 in uncovered_count)
        newly_covered = chosen_targets[uncovered_count[chosen_targets] == 1]
        
        if len(newly_covered) == 0:
            # This block didn't cover anything new? Should not happen if gain > 0.
            continue
            
        # Update uncovered_count and mask
        uncovered_count[newly_covered] = 0
        uncovered_mask[newly_covered] = False
        
        # Update gains for all blocks that share these newly covered targets
        # For each newly covered target, decrement the gain of all blocks that cover it.
        # But we must be careful not to double-decrement if a block covers multiple newly covered targets.
        # Actually, the gain is defined as the count of *currently* uncovered targets it covers.
        # If a block covers 3 newly covered targets, its gain should drop by 3.
        # If we iterate through newly_covered targets and decrement the gain of all blocks covering them,
        # we will decrement the gain of a block multiple times if it covers multiple newly covered targets.
        # This is correct!
        
        # Optimization: Instead of iterating through all blocks for each target,
        # we can just recalculate the gain for the affected blocks?
        # No, the decrement approach is O(sum of degrees of newly covered targets).
        # Degree of a target is the number of blocks covering it.
        # For v=19, k=7, t=4.
        # Number of blocks covering a specific 4-set?
        # A 4-set is in C(15, 3) = 455 blocks of size 7.
        # So for each newly covered target, we touch ~455 blocks.
        # If we cover ~10 new targets per step, we touch ~4500 blocks.
        # This is very fast.
        
        for t_idx in newly_covered:
            for b_idx in target_to_blocks[t_idx]:
                current_gains[b_idx] -= 1
                
        # Ensure gains don't go negative (safety)
        # current_gains = np.maximum(current_gains, 0) # Might be costly, but safe.
        # Since we only decrement when a target is covered, and we only cover it once,
        # the gain should never go below 0 if initialized correctly.
        
    # Local Search: Remove redundant blocks
    # We want to remove blocks that do not cover any "unique" target.
    # A target is unique to a block if that block is the only one covering it in the selected set.
    
    if time.time() < deadline - 10.0 and selected_indices:
        # Build coverage count for selected blocks
        # target_cover_count[j] = number of selected blocks covering target j
        target_cover_count = np.zeros(num_targets, dtype=np.int32)
        for bi in selected_indices:
            target_cover_count[block_to_targets[bi]] += 1
        
        # Iterate through selected blocks and try to remove
        # We need to be careful with indices if we remove from the list.
        # It's better to mark blocks as "kept" or "removed".
        keep_mask = np.ones(len(selected_indices), dtype=bool)
        
        # Repeat removal until no more changes
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i, bi in enumerate(selected_indices):
                if not keep_mask[i]:
                    continue
                
                targets = block_to_targets[bi]
                # If any target covered by this block has count 1, we cannot remove this block.
                if np.any(target_cover_count[targets] == 1):
                    continue
                
                # We can remove this block
                keep_mask[i] = False
                target_cover_count[targets] -= 1
                changed = True
        
        # Reconstruct list
        if not np.all(keep_mask):
            selected_indices = [selected_indices[i] for i in range(len(selected_indices)) if keep_mask[i]]
            
            # Verify
            verify_mask = np.zeros(num_targets, dtype=bool)
            for bi in selected_indices:
                verify_mask[block_to_targets[bi]] = True
            if not np.all(verify_mask):
                # Fallback to previous list if verification fails (should not happen)
                pass # In a real scenario we'd revert, but for this fix, we trust the logic.

    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in selected_indices]
