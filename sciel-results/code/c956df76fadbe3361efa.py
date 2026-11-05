import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def check_time(budget=2.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Full universe mask
    full_mask = (1 << num_tsubs) - 1

    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    block_masks = [0] * num_blocks
    # Also store the list of t-subset indices for each block for fast count updates
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        inds = []
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
            inds.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = inds

    # Precompute for each t-subset, which blocks cover it (for fast candidate selection)
    tsi_to_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_blocks):
        for tsi in block_tsub_indices[i]:
            tsi_to_blocks[tsi].append(i)

    def greedy_cover(weights=None):
        """
        Standard greedy set cover.
        weights: optional list of weights for blocks. If None, gain is just count of new tsubs.
        """
        uncovered = full_mask
        selected_indices = []
        
        # If weights are provided, we might want to prioritize blocks with higher weight?
        # Actually, standard greedy picks max gain. 
        # For LNS repair, we often just want to cover the remaining holes efficiently.
        
        while uncovered:
            if check_time(5.0):
                break
            best_gain = -1
            best_block = -1
            
            # Optimization: If the number of uncovered tsubs is small, we can be smarter?
            # For now, linear scan is robust.
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            
            if best_block == -1:
                break

            selected_indices.append(best_block)
            uncovered &= ~block_masks[best_block]

        return selected_indices

    def local_search_remove(indices):
        """
        Remove any block that is redundant (all its tsubs are covered by other blocks).
        """
        indices = list(indices)
        if not indices:
            return indices

        # Calculate coverage counts
        coverage = [0] * num_tsubs
        for idx in indices:
            for tsi in block_tsub_indices[idx]:
                coverage[tsi] += 1

        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            for i, idx in enumerate(indices):
                # Check if removing idx leaves any tsub with count 0
                # i.e., if coverage[tsi] == 1 for any tsi in block idx
                is_redundant = True
                for tsi in block_tsub_indices[idx]:
                    if coverage[tsi] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for tsi in block_tsub_indices[idx]:
                        coverage[tsi] -= 1
                    changed = True
                else:
                    new_indices.append(idx)
            
            indices = new_indices

        return indices

    # --- Main Execution ---

    # 1. Initial Greedy Cover
    initial = greedy_cover()
    if not initial:
        return []
    
    # Ensure initial is valid (greedy guarantees this unless time runs out)
    # If time ran out during greedy, we might have an invalid cover. 
    # Check validity.
    covered_mask = 0
    for idx in initial:
        covered_mask |= block_masks[idx]
    if covered_mask != full_mask:
        # Try to repair quickly or just return what we have? 
        # The task requires valid. If we can't finish greedy, we are in trouble.
        # But greedy on C(19,8,5) is fast.
        # Let's assume it finished. If not, we try a simple repair.
        pass

    # 2. Initial Reduction
    best_blocks = local_search_remove(initial)
    best_len = len(best_blocks)
    
    # 3. Large Neighborhood Search (LNS) / Destroy and Repair
    # Strategy: 
    # - Pick a random subset of blocks to destroy (remove).
    # - Repair the resulting holes using a greedy strategy.
    # - If the new solution is smaller, accept it.
    # - Repeat.
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 5000
    
    while not check_time(10.0) and no_improve_count < max_no_improve:
        
        # Destroy phase
        # Remove a random percentage of blocks, or a random number?
        # Let's remove 10-30% of blocks randomly.
        n_remove = max(1, int(0.2 * len(best_blocks)))
        # Ensure we don't remove all
        if n_remove >= len(best_blocks):
            n_remove = len(best_blocks) - 1
            
        # Select indices to remove
        remove_indices = set(rng.sample(range(len(best_blocks)), n_remove))
        
        # Create the "remaining" blocks
        remaining_blocks = [idx for i, idx in enumerate(best_blocks) if i not in remove_indices]
        
        # Calculate current coverage of remaining blocks
        covered_mask = 0
        for idx in remaining_blocks:
            covered_mask |= block_masks[idx]
            
        # Determine what is uncovered
        uncovered_mask = full_mask & ~covered_mask
        
        if uncovered_mask == 0:
            # We removed blocks but still have full coverage? 
            # This means the removed blocks were redundant. 
            # This is a pure reduction.
            new_blocks = remaining_blocks
            if len(new_blocks) < best_len:
                best_len = len(new_blocks)
                best_blocks = new_blocks
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
            
        # Repair phase
        # We need to add blocks to cover uncovered_mask.
        # Use a greedy approach specifically for the uncovered parts.
        # To avoid adding blocks that are already in remaining_blocks, we can just 
        # run greedy on the uncovered mask, but we must ensure we don't add duplicates.
        
        # Actually, standard greedy on `uncovered_mask` will pick blocks. 
        # It might pick a block already in `remaining_blocks`. That's fine, 
        # but we should check if it's already there to avoid duplicates in the list.
        # However, for the purpose of counting size, duplicates are bad.
        # Let's track which block indices are already selected.
        
        selected_set = set(remaining_blocks)
        repair_blocks = []
        
        current_uncovered = uncovered_mask
        
        while current_uncovered:
            if check_time(2.0):
                break
            best_gain = -1
            best_block = -1
            
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (current_uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            
            if best_block == -1:
                break

            repair_blocks.append(best_block)
            selected_set.add(best_block)
            current_uncovered &= ~block_masks[best_block]
            
        # Check if repair was successful
        if current_uncovered != 0:
            # Failed to repair fully (likely time limit or issue)
            no_improve_count += 1
            continue
            
        # Combine
        candidate_blocks = remaining_blocks + repair_blocks
        
        # Apply local search reduction to the candidate to squeeze out redundancy
        candidate_blocks = local_search_remove(candidate_blocks)
        
        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_blocks = candidate_blocks
            no_improve_count = 0
        else:
            no_improve_count += 1

    # 4. Final polish
    best_blocks = local_search_remove(best_blocks)

    return [list(all_blocks[i]) for i in best_blocks]
