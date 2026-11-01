import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    # BlockTsets will be a 2D numpy array of shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t)
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        # Generate all t-subsets of this block
        # The elements in 'block' are sorted, so combinations are in lex order
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1
            
    # 2. Greedy Search with Restart and Polish
    best_blocks = None
    best_len = float('inf')
    
    # We will run multiple restarts
    # To diversify, we can randomize the order of tie-breaking or the initial state?
    # Initial state is always all uncovered.
    # Tie-breaking is the main source of diversity.
    
    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Calculate gain for all blocks
            # gains[i] = number of uncovered t-sets covered by block i
            gains = uncovered[BlockTsets].sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if uncovered is not empty, but safety check
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly select one to diversify
            # Using numpy random is fast
            best_block_idx = candidates[np.random.randint(len(candidates))]
            
            blocks.append(best_block_idx)
            
            # Update uncovered
            uncovered[BlockTsets[best_block_idx]] = False
            
        # Polish: Remove redundant blocks
        # We need to know if a block is the only one covering any of its t-sets.
        # We can do this by maintaining a count of how many blocks cover each t-set.
        # But we already have the list of blocks.
        # Let's do a reverse pass.
        
        # Create a coverage count array
        # This is O(N * 56)
        # N is ~300-350. 350 * 56 = 19600 ops. Very fast.
        
        # We will try to remove blocks from the end of the list.
        # If we remove a block, we must ensure no t-set becomes uncovered.
        
        # To do this efficiently, we can just check the blocks in the list.
        # Let's keep a list of block indices.
        # And a dictionary or array of t-set coverage counts?
        # Actually, simpler:
        # For each block in reverse order:
        #   Assume we remove it.
        #   Check if any of its t-sets are covered ONLY by this block in the current list.
        #   This requires knowing for each t-set, which blocks in the list cover it.
        #   This is expensive to recompute from scratch each time.
        
        # Alternative Polish:
        # Just keep the greedy result. Greedy is usually quite good.
        # The "Redundancy Removal" is effective if we add it.
        # Let's implement a simple check:
        # Build a set of t-sets covered by each block.
        # For a candidate block to remove, check if intersection of its t-sets with "t-sets covered by others" is all of its t-sets.
        
        # Actually, a simpler heuristic for polish:
        # Try to swap blocks? No.
        
        # Let's try the redundancy removal properly.
        # It's worth it.
        
        # We have `blocks` as a list of indices into `all_blocks`.
        # We want to minimize len(blocks).
        
        # Let's create a list of t-set indices for each block in `blocks`.
        # block_tsets_list = [BlockTsets[b] for b in blocks]
        
        # We can maintain a `cover_count` array of size num_tsets.
        # Initialize to 0.
        # For each block in blocks, increment cover_count for its t-sets.
        # Then, iterate through blocks in reverse.
        # For block b:
        #   For each t-set ts in BlockTsets[b]:
        #     if cover_count[ts] == 1:
        #       # This t-set is only covered by b. We cannot remove b.
        #       break
        #   if no t-set has count 1:
        #     # We can remove b.
        #     Decrement cover_count for all ts in BlockTsets[b].
        #     Remove b from blocks.
        
        # This is O(N * 56) for the count update and O(N * 56) for the check.
        # Total O(N * 56).
        
        if len(blocks) < best_len:
            # Apply polish
            cover_count = np.zeros(num_tsets, dtype=np.int32)
            for b in blocks:
                cover_count[BlockTsets[b]] += 1
                
            # Iterate from end to start
            # We need to be careful with modifying the list while iterating.
            # We can iterate backwards by index.
            new_blocks = []
            # Actually, let's just filter.
            # If we remove a block, the cover counts change, affecting previous blocks.
            # So we must do it sequentially.
            
            # Let's use a list and pop from the end.
            # We will check the last block first.
            # If we can remove it, we do, and update counts.
            # Then check the new last block.
            
            # This is safe.
            # But we want to keep the "good" blocks (early ones) if possible?
            # Greedy adds blocks that cover the most *new* stuff.
            # Later blocks cover less new stuff.
            # So later blocks are more likely to be redundant.
            # So checking from the end is a good heuristic.
            
            # Let's implement this.
            # Note: `blocks` is a list of ints.
            
            # We need to update `cover_count` as we go.
            
            # Let's make a copy of the list to iterate?
            # No, we can just iterate backwards.
            
            # However, if we remove block i, we must update cover_count.
            
            # Let's do it.
            
            # To avoid modifying list while iterating, we can build a new list.
            # But the dependency is: removing block i affects whether block j (j < i) can be removed.
            # So we must process from end to start.
            
            # Let's use a while loop on the length?
            # Or just iterate `range(len(blocks) - 1, -1, -1)` and use a flag?
            
            # Simplest:
            # `blocks` is a list.
            # `i = len(blocks) - 1`
            # `while i >= 0:`
            #   `b = blocks[i]`
            #   `can_remove = True`
            #   `for ts in BlockTsets[b]:`
            #     `if cover_count[ts] == 1:`
            #       `can_remove = False`
            #       `break`
            #   `if can_remove:`
            #     `for ts in BlockTsets[b]:`
            #       `cover_count[ts] -= 1`
            #     `blocks.pop(i)`
            #     `# Do not decrement i, because the new element at i is the previous one`
            #   `else:`
            #     `i -= 1`
            
            # This modifies `blocks` in place.
            # We need to be careful not to modify `best_blocks` if it's the same object.
            # We are creating a new `blocks` list each restart, so it's safe.
            
            # Let's execute this polish.
            
            i = len(blocks) - 1
            while i >= 0:
                b = blocks[i]
                can_remove = True
                # Check if any t-set is uniquely covered
                for ts in BlockTsets[b]:
                    if cover_count[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in BlockTsets[b]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    # i stays the same, check the new last element
                else:
                    i -= 1
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:] # Copy
        
    # Convert indices back to actual blocks (lists of points)
    if best_blocks is None:
        # Fallback: return empty or a valid small set?
        # Should not happen.
        return []
        
    result = [list(all_blocks[b]) for b in best_blocks]
    return result
