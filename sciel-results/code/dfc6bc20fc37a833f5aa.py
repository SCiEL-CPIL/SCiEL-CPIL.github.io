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
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Full universe mask: all bits set to 1
    full_mask = (1 << num_tsubs) - 1

    # Precompute all blocks and their corresponding t-subset masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if t-subset j is in block i
    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # --- Greedy Set Cover using Bitmasks ---
    def greedy_cover():
        """
        Standard greedy algorithm for set cover.
        At each step, pick the block that covers the most uncovered t-subsets.
        """
        uncovered = full_mask
        selected_indices = []
        remaining_blocks = set(range(num_blocks))
        
        # Optimization: If we just want a valid cover, standard greedy is good.
        # To speed up, we can pre-sort or use heuristics, but with 75k blocks,
        # a simple loop over remaining blocks to find max gain is O(N * M) per step.
        # N ~ 75k, M ~ 300 steps. 75k * 300 = 22.5M operations.
        # Bitwise operations on 11k bit ints are fast in Python (C implementation).
        
        while uncovered:
            if check_time(10.0):
                break
            best_gain = -1
            best_block = -1
            
            # To speed this up, we can sample or use a heap, but for 300 iterations,
            # a linear scan of 75k items is ~22M big-int ops. Might be slow in Python.
            # Let's optimize by only checking blocks that have potential gain.
            # Actually, let's just iterate. 75k iterations * 300 steps = 22.5M.
            # Each iteration is: gain = bin(uncovered & block_masks[i]).count('1')
            # This is expensive. 
            # Alternative: Use NumPy? No, ints are arbitrary precision.
            # Better: Pre-compute nothing, but use a heuristic.
            # Let's try a randomized greedy or a limited candidate set.
            # Or, just do it. 22M simple bitwise ANDs and popcounts.
            # Python's int.bit_count() is O(1) effectively (word operations).
            
            for i in remaining_blocks:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                # Early break if we found a perfect cover for remaining? Unlikely.
                
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            remaining_blocks.remove(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected_indices

    # --- Local Search: 1-Opt Removal ---
    def local_search_remove(indices):
        """
        Given a list of block indices that form a cover, try to remove any block
        that is redundant (i.e., its removal does not change the union of masks).
        """
        indices = list(indices)
        if not indices:
            return indices
            
        # Calculate union of all masks
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
            
        changed = True
        passes = 0
        while changed and not check_time(5.0):
            changed = False
            passes += 1
            new_indices = []
            for i, idx in enumerate(indices):
                # If we remove idx, does the union change?
                # New union = union_mask ^ (block_masks[idx] & ~ (union of others))?
                # Simpler: (union_mask & ~block_masks[idx]) == union_mask?
                # No, that's not right. 
                # We need to check if block_masks[idx] is a subset of the union of the OTHER blocks.
                # Let U = union of all.
                # Let U_i = union of all except i.
                # If block_masks[i] & ~U_i == 0, then i is redundant.
                # U_i = U & ~block_masks[i] is NOT correct if block i is the only one covering some bits.
                # Actually, U_i is the union of all masks except i.
                # We can compute U_i by iterating? That's O(N) per check. Total O(N^2).
                # 300^2 = 90k operations. Very fast.
                
                # Calculate union of all others
                others_union = 0
                for j, other_idx in enumerate(indices):
                    if j != i:
                        others_union |= block_masks[other_idx]
                        
                # Check if idx adds anything
                if (block_masks[idx] & ~others_union) == 0:
                    # Redundant, skip it
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices = new_indices
            
        return indices

    # --- Simulated Annealing / Randomized Local Search ---
    # Since Greedy + 1-Opt might get stuck, we add a randomization phase.
    # We will perform "swap" moves: remove one block, add another, if it maintains coverage.
    
    def randomized_local_search(initial_indices, seed=42):
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices
            
        # Precompute union
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
            
            # Move: Remove a random block, then try to add a block that covers the lost parts?
            # Or simpler: Try to replace a block with a "better" one?
            # Actually, since we want to minimize count, "Removal" is the only direction that improves count.
            # "Swap" keeps count same.
            # So, let's stick to trying to remove blocks.
            # If we can't remove any, we are at a local minimum for 1-Opt.
            # To escape, we can do a "2-Opt": Remove 2 blocks, add 1 block?
            # Or, Random Restart: Take current set, remove a random block (breaking coverage), 
            # then greedily repair coverage.
            
            # Let's try "Repair Strategy":
            # 1. Pick a random block to remove.
            # 2. If coverage is broken, greedily add blocks to restore coverage.
            # 3. If the new size is <= old size, accept.
            
            if len(indices) < 2:
                break
                
            remove_idx = rng.randint(0, len(indices) - 1)
            removed_block = indices[remove_idx]
            
            # Tentative removal
            new_indices = [x for x in indices if x != removed_block]
            
            # Calculate new union
            new_union = 0
            for idx in new_indices:
                new_union |= block_masks[idx]
                
            missing = full_mask & ~new_union
            
            if missing == 0:
                # We found a redundant block! (Should have been caught by 1-opt, but maybe due to order?)
                # Actually, 1-opt checks if block is subset of others. 
                # If missing==0, it is redundant.
                indices = new_indices
                union_mask = new_union
                current_len = len(indices)
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue
            
            # Coverage is broken. Try to repair.
            # Greedily add blocks to cover 'missing'.
            # To avoid adding too many, limit the repair depth.
            repair_depth = 3
            added_blocks = []
            temp_missing = missing
            
            for _ in range(repair_depth):
                if temp_missing == 0:
                    break
                # Find best block to add
                best_gain = -1
                best_add = -1
                # Sample a subset of blocks to speed up? Or all?
                # All is 75k. 3 * 75k = 225k ops per repair attempt.
                # 50k iterations * 225k = 11B ops. Too slow.
                # We must sample or pre-filter.
                
                # Pre-filter: Only consider blocks that cover at least one missing t-subset.
                # We can't easily filter without iterating.
                # Let's just sample 1000 random blocks.
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
                
            # Check if we successfully repaired
            if temp_missing == 0:
                # New set is new_indices + added_blocks
                new_size = len(new_indices) + len(added_blocks)
                if new_size < current_len:
                    # Accept
                    indices = new_indices + added_blocks
                    # Recalculate union (should be full)
                    union_mask = full_mask
                    current_len = new_size
                    if current_len < best_len:
                        best_len = current_len
                        best_indices = list(indices)
                elif new_size == current_len and rng.random() < 0.1:
                    # Accept swap (neutral move) to escape local minima
                    indices = new_indices + added_blocks
                    union_mask = full_mask
            # If not repaired, reject (keep old indices)
            
            T *= alpha
            
        return best_indices

    # Main Execution
    # 1. Greedy Initialization
    initial = greedy_cover()
    
    if not initial:
        # Fallback if greedy failed (shouldn't happen)
        return []
        
    # 2. 1-Opt Reduction
    reduced = local_search_remove(initial)
    
    best_blocks = list(reduced)
    best_len = len(best_blocks)
    
    # 3. Randomized Local Search (Repair-based)
    # Run a few times with different seeds
    for seed in range(5):
        if check_time(15.0):
            break
        result = randomized_local_search(reduced, seed=seed * 100 + 1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result
            
    # 4. Final 1-Opt pass on the best result
    best_blocks = local_search_remove(best_blocks)
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_blocks]

# To ensure the function is available at module level as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program defines construct(v, k, t).
# I have defined construct above.
