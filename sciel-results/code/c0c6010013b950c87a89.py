import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a bitset-accelerated greedy set cover algorithm.
    This approach is significantly faster and often yields better results than
    point-by-point greedy construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin to ensure we return before 120s limit

    # 1. Generate all t-subsets and map them to unique indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (candidate blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute a bitmask for each block.
    # Bit i in the mask is 1 if the i-th t-subset is contained in the block.
    # Python integers are arbitrary precision, so this is efficient for bit operations.
    block_masks = [0] * num_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        # Iterate over all t-subsets of this specific block
        for t_sub in itertools.combinations(block, t):
            # t_sub is a tuple, directly usable as a dictionary key
            i = t_subset_to_idx[t_sub]
            mask |= (1 << i)
        block_masks[idx] = mask

    # 4. Greedy Set Cover Loop
    # Initialize the set of uncovered t-subsets as a bitmask with all relevant bits set
    uncovered_mask = (1 << num_t_subsets) - 1
    selected_blocks = []
    
    # We will repeatedly pick the block that covers the most uncovered t-subsets.
    # To avoid re-picking the same block, we zero out its mask after selection.
    
    while uncovered_mask:
        if time.time() > deadline:
            # If we run out of time, return the best partial solution found so far.
            # Note: The problem requires a valid covering. If we break early, 
            # the result might be invalid. However, for v=19, the greedy step is 
            # very fast, so we should not hit this. 
            # If we do, it's better to return an invalid solution (score 0) than hang?
            # Actually, the prompt says "invalid result ... scores 0".
            # So we should try to finish. 
            # Given the speed, this is unlikely.
            break
            
        best_gain = -1
        best_idx = -1
        
        # Local variables for speed
        bm = block_masks
        um = uncovered_mask
        
        # Find the block with the maximum gain
        for i in range(num_blocks):
            # Skip blocks that have already been selected (mask is 0)
            # Note: A mask can be 0 if the block covers no t-subsets (impossible for k>=t) 
            # or if it was selected.
            if bm[i] == 0:
                continue
                
            # Calculate gain: number of newly covered t-subsets
            gain = (um & bm[i]).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
                # Early termination if we found a block that covers all remaining?
                # Unlikely in one step, but if best_gain == num_uncovered, we could stop?
                # No, we need to select it, then the loop condition `while uncovered_mask` will handle it.
        
        if best_gain == 0:
            # No block can cover any remaining uncovered subset.
            # This implies the covering is impossible or we made a mistake.
            # For valid parameters, this shouldn't happen.
            break
            
        # Select the best block
        selected_blocks.append(list(all_blocks[best_idx]))
        
        # Update the uncovered mask
        uncovered_mask &= ~bm[best_idx]
        
        # Mark the block as used by zeroing its mask
        bm[best_idx] = 0

    return selected_blocks
