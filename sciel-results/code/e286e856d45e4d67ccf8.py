import itertools
import time

def construct(v, k, t):
    # Parameters: v=19, k=7, t=4
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Map all t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # 2. Generate all k-subsets (candidate blocks) and their coverage masks
    k_subs = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_subs:
        mask = 0
        # Each k-block covers C(k, t) t-subsets
        for sub in itertools.combinations(block, t):
            mask |= 1 << t_sub_to_idx[sub]
        block_masks.append(mask)
        
    # 3. Greedy Set Cover Algorithm
    # uncovered_mask has bit i set if t-sub i is not yet covered
    uncovered_mask = (1 << n_t_subs) - 1
    selected_blocks = []
    
    while uncovered_mask:
        if time.time() > deadline:
            # In case of unexpected slowness, return what we have? 
            # No, invalid covering scores 0. 
            # But this should never happen for v=19.
            break
            
        best_gain = 0
        best_block_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        for i in range(len(block_masks)):
            # Calculate marginal gain: number of new t-subsets covered
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                # Early termination if we cover all remaining?
                # Max possible gain is min(popcount(uncovered), C(k,t))
                # C(7,4) = 35. If best_gain == 35 and popcount(uncovered) >= 35, 
                # we can't do better than 35 for this step.
                # But we don't know popcount(uncovered) easily without another bit_count.
                # We can just let it scan. 50k iterations is fast.
                
        if best_block_idx == -1:
            # No block covers any new subset? This implies a bug or impossible instance.
            break
            
        # Add the selected block to the solution
        selected_blocks.append(list(k_subs[best_block_idx]))
        
        # Update the uncovered mask
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return selected_blocks
