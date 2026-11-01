import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Pre-compute the t-subset indices for each block
    # block_subsets[i] is a list of t_subset_indices covered by block i
    block_subsets = []
    for block in all_blocks:
        block_set = set(block)
        covered_indices = []
        for ts in all_t_subsets:
            if all(x in block_set for x in ts):
                covered_indices.append(t_subset_index[ts])
        block_subsets.append(covered_indices)
        
    # Convert to NumPy array for fast vectorized operations
    # We need a uniform length for the array. Max t-subsets per block is C(k,t).
    max_t_per_block = len(block_subsets[0])
    # Actually, all blocks have exactly C(k,t) t-subsets.
    block_subsets_np = np.array(block_subsets, dtype=np.int32)
    
    best_blocks = []
    best_count = float('inf')
    
    # Multiple restarts to find a better covering
    # Each restart uses a different random tie-breaking strategy
    while time.time() < deadline:
        covered = np.zeros(num_t_subsets, dtype=bool)
        remaining_blocks = set(range(num_blocks))
        current_blocks = []
        
        while True:
            if time.time() > deadline:
                break
            
            if not remaining_blocks:
                break
                
            # Calculate gains for all remaining blocks
            # We can't easily index a subset of rows in a pre-allocated array without a mask or list comprehension
            # Since num_blocks is ~50k, and we do this ~150 times, 
            # creating a list of indices for remaining blocks is fine.
            
            # Optimization: Only calculate gains for blocks that might have gain > 0?
            # It's hard to know without checking.
            # Let's just compute for all remaining.
            
            # To speed up, we can maintain a list of active block indices
            # But NumPy fancy indexing is fast.
            
            active_indices = list(remaining_blocks)
            if not active_indices:
                break
                
            # Get the t-subset indices for active blocks
            active_block_subsets = block_subsets_np[active_indices]
            
            # Check which of these are NOT covered
            # active_block_subsets is (N, 35)
            # covered is (3876,)
            # covered[active_block_subsets] is (N, 35) of bools
            not_covered = ~covered[active_block_subsets]
            
            # Sum over axis 1 to get gain for each block
            gains = not_covered.sum(axis=1)
            
            # Find the block with maximum gain
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # No block covers any new subset? This shouldn't happen if we haven't finished
                # unless we are in a bad state or covered is full
                if not np.all(covered):
                    # Fallback: pick any block? Or break?
                    # If we can't cover more, we are stuck. 
                    # But greedy set cover on a valid universe should always find a block with gain > 0 
                    # until covered is full, because every uncovered subset is in *some* block.
                    # Wait, if we removed blocks? No, we don't remove blocks from the pool, we just pick one.
                    # The pool is fixed. So if there is an uncovered subset, there is a block containing it.
                    # So max_gain should be > 0.
                    pass
                break

            # Find all blocks with max_gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to break ties differently in each restart
            # Or pick the first one. Randomization helps escape local optima in greedy.
            chosen_local_idx = candidates[np.random.randint(len(candidates))]
            chosen_global_idx = active_indices[chosen_local_idx]
            
            current_blocks.append(all_blocks[chosen_global_idx])
            
            # Update covered
            # We need to mark the t-subsets of this block as covered
            # We can use the pre-computed indices
            covered[block_subsets_np[chosen_global_idx]] = True
            
            # Remove this block from remaining? 
            # In set cover, we don't strictly need to remove it, but it has gain 0 now.
            # Keeping it in the pool allows it to be considered (with gain 0) but that's wasted computation.
            # Removing it reduces the size of active_indices.
            remaining_blocks.remove(chosen_global_idx)
            
            if np.all(covered):
                break
        
        # End of one restart
        if len(current_blocks) < best_count and np.all(covered):
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
        # If we hit the deadline, break out of restart loop
        if time.time() > deadline:
            break
            
    return [list(b) for b in best_blocks]
