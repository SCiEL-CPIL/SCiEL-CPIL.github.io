import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute t-subsets and their indices
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}
    
    # Precompute all k-subsets (blocks) and their masks
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    block_masks = []
    # Also store the list of t-subset indices for each block for faster redundancy checks
    block_tsub_indices = []
    for comb in all_k_subsets:
        mask = 0
        indices = []
        for sub in itertools.combinations(comb, t):
            idx = tsub_to_bit[sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks.append(mask)
        block_tsub_indices.append(indices)
        
    full_mask = (1 << n_tsub) - 1
    
    # Helper: Greedy construction/repair
    def greedy_repair(current_blocks, uncovered_mask):
        """
        current_blocks: set of block indices
        uncovered_mask: bitmask of t-subsets not covered by current_blocks
        Returns: (new_blocks_set, new_uncovered_mask)
        """
        blocks = set(current_blocks)
        unc = uncovered_mask
        
        while unc and time.time() < deadline:
            best_gain = 0
            best_idx = -1
            
            # Iterate through all blocks not in current set
            # Optimization: If we have many blocks, this is slow. 
            # But n_ksub is ~50k. 
            # We can break early if we find a block that covers all remaining? Unlikely.
            for i in range(n_ksub):
                if i in blocks:
                    continue
                g = (block_masks[i] & unc).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_idx = i
                    if best_gain == unc.bit_count(): # Covers all remaining
                        break
            
            if best_gain == 0:
                break # Should not happen if problem is solvable, but safety
                
            blocks.add(best_idx)
            unc &= ~block_masks[best_idx]
            
        return blocks, unc

    # Helper: Reduce redundant blocks
    def reduce_blocks(blocks):
        """
        Removes blocks that are redundant.
        Returns new set of blocks.
        """
        # Calculate cover counts
        cover_count = [0] * n_tsub
        for b in blocks:
            for idx in block_tsub_indices[b]:
                cover_count[idx] += 1
        
        new_blocks = set()
        # Try to remove each block
        # To make it more effective, remove in random order or multiple passes
        # We do a single pass first, then repeat if time permits
        for b in list(blocks):
            # Check if any t-subset in this block is covered ONLY by this block
            redundant = True
            for idx in block_tsub_indices[b]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for idx in block_tsub_indices[b]:
                    cover_count[idx] -= 1
            else:
                new_blocks.add(b)
                
        return new_blocks

    # Initial Solution
    current_blocks, _ = greedy_repair(set(), full_mask)
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = set(current_blocks)
    best_size = len(best_blocks)
    
    # Local Search Loop: Destroy and Repair
    while time.time() < deadline:
        # Perturb: Remove some random blocks
        # Number of blocks to remove: 1 to 3
        n_remove = random.randint(1, 3)
        if len(current_blocks) <= n_remove:
            # Restart if too small to perturb meaningfully
            current_blocks, _ = greedy_repair(set(), full_mask)
            current_blocks = reduce_blocks(current_blocks)
            continue
            
        # Select blocks to remove
        items = list(current_blocks)
        random.shuffle(items)
        removed = set(items[:n_remove])
        
        # Calculate uncovered mask after removal
        unc_mask = 0
        # We need to know which t-subsets are left uncovered.
        # It's faster to recalculate cover counts or just check.
        # Since we removed few blocks, we can check which t-subsets are now uncovered.
        # A t-subset is uncovered if its cover count drops to 0.
        
        # Efficient way: 
        # 1. Count coverage of current_blocks (before removal) is not stored globally.
        # Let's just compute the mask of covered t-subsets for the remaining blocks.
        # covered_mask = OR of masks of (current_blocks - removed)
        # This is O(N_blocks) which is 50k. Acceptable.
        
        remaining = current_blocks - removed
        covered_mask = 0
        for b in remaining:
            covered_mask |= block_masks[b]
            
        unc_mask = full_mask & ~covered_mask
        
        # Repair
        repaired_blocks, final_unc = greedy_repair(remaining, unc_mask)
        
        if final_unc != 0:
            # Repair failed? Should not happen. But if it does, skip.
            continue
            
        # Reduce
        repaired_blocks = reduce_blocks(repaired_blocks)
        
        # Update best
        if len(repaired_blocks) < best_size:
            best_size = len(repaired_blocks)
            best_blocks = set(repaired_blocks)
            
        # Update current (accept if better or equal, or with some probability)
        # For simple ILS, we usually accept if better. 
        # To escape local minima, we might accept worse solutions with prob exp(-delta/T).
        # But since we want min size, and repair is greedy, it's hard to get worse.
        # Let's just set current to the repaired one.
        current_blocks = repaired_blocks

    # Final Output
    result = [list(all_k_subsets[idx]) for idx in best_blocks]
    return result
