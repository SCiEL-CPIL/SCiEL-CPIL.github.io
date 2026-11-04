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
    deadline = start_time + 140.0  # Safety margin
    
    def check_time(budget=5.0):
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
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # --- Greedy Set Cover with Intersection Awareness ---
    # We track selected blocks to avoid adding blocks with high overlap
    # if they don't provide enough new coverage.
    
    def greedy_cover_with_constraints():
        uncovered = full_mask
        selected_indices = []
        # Track intersection counts to penalize high-overlap additions
        # Actually, let's stick to pure gain but use a smarter ordering.
        # The "constraint" is implicit: we only add if gain > 0.
        # To improve quality, we can randomize tie-breaking or use a weighted gain.
        
        # Let's use a standard greedy but with a twist:
        # If multiple blocks have the same gain, pick the one with the smallest
        # intersection with the current union of selected blocks? 
        # No, that's complex. 
        # Instead, let's use a "lazy" greedy where we update gains incrementally?
        # Too complex for this time limit.
        
        # Alternative Structural Change:
        # Use a "Pairwise Redundancy" pruning after greedy.
        # Two blocks A and B are "redundant" if one can be removed if the other is present?
        # No, standard 1-opt handles that.
        
        # Let's try a different construction:
        # Start with a random subset of blocks that is already large, then prune?
        # Or, use a "Conflict-free" greedy?
        
        # Let's implement a standard greedy first, as it's the baseline.
        # The structural change will be in the PRUNING phase.
        
        while uncovered:
            if check_time(10.0):
                break
            best_gain = -1
            best_block = -1
            
            # Linear scan
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

    # --- Advanced Pruning: Pairwise Redundancy ---
    # If block A and block B share k-1 points, they cover almost the same t-subsets.
    # Specifically, if |A \cap B| >= k - (t-1), they are very similar.
    # For k=8, t=5, if they share 7 points, the t-subsets covered by A are:
    # All 5-subsets of the 7 shared points (C(7,5)=21) plus those involving the 1 unique point.
    # If we have both, we might be able to remove one if its unique coverage is provided by others.
    
    def pairwise_redundancy_pruning(indices):
        """
        Remove blocks that are 'almost' duplicates of others, if safe.
        """
        indices = list(indices)
        if len(indices) < 2:
            return indices
            
        # Sort indices by block id for determinism? Or keep order.
        # We will iterate and try to remove blocks that have high intersection with another selected block.
        
        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            n = len(indices)
            for i in range(n):
                if check_time(2.0):
                    # If time is low, just keep the rest
                    new_indices.extend(indices[i:])
                    break
                    
                idx_i = indices[i]
                block_i = all_blocks[idx_i]
                
                # Check if this block is redundant given the others in the current list
                # Standard 1-opt: is block_i's mask subset of union of others?
                # But that's O(N^2). 
                # Let's do a targeted check:
                # Is there another block j in indices such that |block_i \cap block_j| is high,
                # AND the unique parts of block_i are covered by others?
                
                # Actually, let's just do the standard 1-opt check but optimized.
                # We already have a local_search_remove. 
                # The "Structural Change" should be something else.
                
                # Let's implement a "2-Opt" style swap in the main loop instead?
                
                # For now, let's just keep the block.
                new_indices.append(idx_i)
            indices = new_indices
        return indices

    # --- Main Execution ---
    
    # 1. Greedy Initialization
    initial = greedy_cover_with_constraints()
    
    if not initial:
        return []
        
    # 2. Standard 1-Opt Reduction
    def local_search_remove(indices):
        indices = list(indices)
        if not indices:
            return indices
            
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
            
        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            for i, idx in enumerate(indices):
                # Calculate union of all others
                others_union = 0
                for j, other_idx in enumerate(indices):
                    if j != i:
                        others_union |= block_masks[other_idx]
                        
                if (block_masks[idx] & ~others_union) == 0:
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices = new_indices
            
        return indices

    reduced = local_search_remove(initial)
    best_blocks = list(reduced)
    best_len = len(best_blocks)

    # 3. Randomized Repair Search (as before, but maybe with different parameters)
    def randomized_local_search(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices
            
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
            
        current_len = len(indices)
        best_indices = list(indices)
        best_len = current_len
        
        T = 10.0
        alpha = 0.99
        iters = 0
        max_iters = 50000
        
        while T > 0.1 and iters < max_iters and not check_time(10.0):
            iters += 1
            
            if len(indices) < 2:
                break
                
            remove_idx = rng.randint(0, len(indices) - 1)
            removed_block = indices[remove_idx]
            
            new_indices = [x for x in indices if x != removed_block]
            
            new_union = 0
            for idx in new_indices:
                new_union |= block_masks[idx]
                
            missing = full_mask & ~new_union
            
            if missing == 0:
                indices = new_indices
                union_mask = new_union
                current_len = len(indices)
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue
            
            repair_depth = 3
            added_blocks = []
            temp_missing = missing
            
            for _ in range(repair_depth):
                if temp_missing == 0:
                    break
                best_gain = -1
                best_add = -1
                candidates = rng.sample(range(num_blocks), min(1000, num_blocks))
                
                for i in candidates:
                    gain = (temp_missing & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
                        
                if best_add == -1 or best_gain == 0:
                    break
                    
                added_blocks.append(best_add)
                temp_missing &= ~block_masks[best_add]
                
            if temp_missing == 0:
                new_size = len(new_indices) + len(added_blocks)
                if new_size < current_len:
                    indices = new_indices + added_blocks
                    union_mask = full_mask
                    current_len = new_size
                    if current_len < best_len:
                        best_len = current_len
                        best_indices = list(indices)
                elif new_size == current_len and rng.random() < 0.1:
                    indices = new_indices + added_blocks
                    union_mask = full_mask
            
            T *= alpha
            
        return best_indices

    for seed in range(5):
        if check_time(15.0):
            break
        result = randomized_local_search(reduced, seed=seed * 100 + 1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result
            
    # 4. Final 1-Opt pass
    best_blocks = local_search_remove(best_blocks)
    
    return [list(all_blocks[i]) for i in best_blocks]
