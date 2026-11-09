import itertools
import random

def construct(v, k, t):
    # 1. Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Generate all k-subsets and their bitmasks
    k_subs = list(itertools.combinations(range(v), k))
    block_data = [] # list of (block_tuple, mask_int)
    
    for block in k_subs:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))
        
    # 3. Greedy Set Cover
    uncovered_mask = (1 << num_t_subs) - 1
    solution = []
    solution_masks = []
    
    while uncovered_mask:
        best_gain = 0
        best_block = None
        best_mask = None
        
        # Iterate through all blocks to find the one with max gain
        # Optimization: If num_t_subs is large, this loop is the bottleneck.
        # For v=24, num_t_subs=2024. 1<<2024 is a big int, but bit_count is fast.
        for block, mask in block_data:
            gain = (uncovered_mask & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_mask = mask
                
        if best_gain == 0:
            # Should not happen if covering is possible
            break
            
        solution.append(list(best_block))
        solution_masks.append(best_mask)
        uncovered_mask &= ~best_mask
        
    # 4. Local Search: Prune redundant blocks
    # Maintain count of how many blocks cover each t-subset
    counts = [0] * num_t_subs
    for mask in solution_masks:
        # Iterate through bits in mask? Or just iterate through t_subs in block?
        # It's faster to iterate through the block's t_subs.
        pass # We need the block's t_subs. Let's re-calculate or store.
        
    # Better pruning:
    # Re-calculate counts from solution
    for block in solution:
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            counts[idx] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(solution)):
            block = solution[i]
            # Tentatively remove
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                counts[idx] -= 1
                
            # Check if any count is 0
            if min(counts) > 0:
                # Valid removal
                solution.pop(i)
                changed = True
                break # Restart loop to check other blocks again
            else:
                # Restore
                for t_sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[t_sub]
                    counts[idx] += 1
                    
    return solution
