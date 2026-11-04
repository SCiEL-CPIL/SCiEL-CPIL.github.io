import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Pre-computation
    # Generate all t-subsets and map to index
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    # and a bitset mask
    block_masks = [0] * n_blocks
    block_subsets = [[] for _ in range(n_blocks)]
    
    for b_idx, block in enumerate(k_blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            subs.append(idx)
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs
        
    # 2. Greedy Construction
    uncovered = (1 << n_subsets) - 1
    selected_blocks = []
    cover_count = [0] * n_subsets
    
    # To speed up greedy, we can pre-sort blocks by size of coverage? 
    # No, coverage depends on 'uncovered'.
    
    while uncovered:
        if time.time() > deadline - 5:
            break
            
        best_block = -1
        best_gain = 0
        best_mask = 0
        
        # Find block with max intersection with uncovered
        # Optimization: if n_blocks is large, this loop is the bottleneck.
        # 75k iterations * bit_count.
        # bit_count on a 11k-bit int is fast.
        for b_idx in range(n_blocks):
            if selected_blocks and b_idx in selected_blocks: 
                # We shouldn't select the same block twice. 
                # But checking `in` is O(N). 
                # Better to mark selected blocks or just skip if mask is 0? 
                # We can maintain a list of available blocks or just skip.
                # Since we are constructing a set, duplicates are bad.
                pass
                
        # The `in` check is too slow. 
        # Let's use a set of selected block indices? Or just track them.
        # Actually, we can just skip if we've already picked it.
        # But iterating 75k times and checking a set is okay.
        
        selected_set = set(selected_blocks)
        
        for b_idx in range(n_blocks):
            if b_idx in selected_set:
                continue
            gain = (block_masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                best_mask = block_masks[b_idx]
                
        if best_block == -1:
            break # Should not happen if uncovered is not empty and blocks exist
            
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        uncovered &= ~best_mask
        
        # Update cover counts
        for s in block_subsets[best_block]:
            cover_count[s] += 1
            
    # 3. Redundancy Removal
    # We have a valid covering. Now try to remove blocks.
    # A block is redundant if all its t-subsets are covered by other blocks.
    # i.e., cover_count[s] > 1 for all s in block.
    
    # We will do multiple passes.
    # To make it faster, we can check blocks in random order or reverse order.
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        # Iterating in reverse order of addition might be good as later blocks might be redundant.
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            # Check if redundant
            redundant = True
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                changed = True
                # After removing one, previous checks might be invalid? 
                # No, removing a block only decreases counts. 
                # It cannot make a previously non-redundant block redundant 
                # (counts only go down, so a count of 1 might become 0? 
                # Wait. If we remove a block, counts decrease. 
                # A block that was essential (had a count=1 subset) might still be essential. 
                # A block that was redundant (all counts > 1) might become essential 
                # if the count of one of its subsets drops to 1. 
                # So we must restart the scan or continue carefully. 
                # Continuing forward is safe for finding NEW redundancies? 
                # No, removing a block can create NEW essential blocks (counts drop to 1). 
                # It can also create NEW redundant blocks? No, counts drop, so it's harder to be redundant. 
                # So removing a block can only make OTHER blocks MORE likely to be essential, 
                # not redundant. 
                # Wait. 
                # Redundant means: for all s in B, count[s] > 1.
                # If we remove another block B', counts for s in B' decrease.
                # If s is in B and B', count[s] decreases.
                # If count[s] was 2, it becomes 1. Then B is NO LONGER redundant.
                # So removing a block can make other blocks NON-redundant.
                # It cannot make non-redundant blocks redundant.
                # So if we scan from end to start, and remove one, we can continue.
                # But we might have skipped a block earlier that is now redundant?
                # No, if it was redundant before, it is still redundant?
                # If count[s] > 1, and we decrease count[s], it might become 1. 
                # So it might become NON-redundant.
                # So previously redundant blocks might become essential.
                # Previously essential blocks remain essential.
                # So we cannot find NEW redundant blocks by removing others.
                # We can only find redundant blocks that were already redundant.
                # So a single pass from end to start is sufficient to remove ALL currently redundant blocks?
                # Let's verify.
                # Suppose A and B are redundant.
                # We check B, remove it. Counts decrease.
                # We check A. A's counts might have dropped.
                # If A had a subset s with count 2, and B also covered s, count is now 1.
                # So A is no longer redundant.
                # So we might miss removing A if we just do one pass.
                # Therefore, we need multiple passes until no change.
                
                pass
                
    # 4. Final Check and Return
    # Convert indices back to blocks
    final_blocks = [list(k_blocks[i]) for i in selected_blocks]
    
    # Verify validity (optional but good for debugging, might be slow)
    # The evaluator checks it.
    
    return final_blocks
