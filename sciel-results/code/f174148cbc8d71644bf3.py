import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Pre-compute all t-subsets and map them to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Pre-compute all k-blocks and their bitmask of t-subsets
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # mask[b] has bit i set if t_sub_to_id[t_subs[i]] is in k_blocks[b]
    block_masks = [0] * num_k_blocks
    for b_idx, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= 1 << t_sub_to_id[t_sub]
        block_masks[b_idx] = mask
    
    # Initial uncovered mask: all t-subsets are uncovered
    full_mask = (1 << num_t_subs) - 1
    
    best_num_blocks = float('inf')
    best_blocks = None
    
    # Run multiple restarts with random tie-breaking
    num_restarts = 500  # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = full_mask
        blocks_used = []
        
        while uncovered:
            max_gain = 0
            candidates = []
            
            # Find blocks that cover the most uncovered subsets
            for b_idx in range(num_k_blocks):
                gain = (uncovered & block_masks[b_idx]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [b_idx]
                elif gain == max_gain:
                    candidates.append(b_idx)
            
            if not candidates:
                break # Should not happen if uncovered is not empty
            
            # Randomly pick one from the best candidates to diversify
            chosen_block_idx = random.choice(candidates)
            blocks_used.append(chosen_block_idx)
            uncovered &= ~block_masks[chosen_block_idx]
            
        # Local Search: Pruning redundant blocks
        # We represent the current solution as a list of block indices
        # To check redundancy, we need to know the coverage count of each t-subset
        # Or simpler: for each block B, check if all its t-subsets are covered by OTHER blocks.
        
        # Efficient pruning:
        # 1. Count coverage of each t-subset by the current set of blocks.
        # 2. If a block B has a t-subset with count 1, B is essential.
        # 3. If all t-subsets in B have count > 1, B is redundant.
        
        # We do this iteratively.
        current_blocks = list(blocks_used)
        
        while True:
            if time.time() > deadline:
                break
                
            # Calculate coverage counts
            coverage = [0] * num_t_subs
            for b_idx in current_blocks:
                m = block_masks[b_idx]
                # Iterate bits set in m
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    coverage[idx] += 1
                    mm ^= lsb
            
            # Check for redundant blocks
            removed_any = False
            new_current_blocks = []
            for b_idx in current_blocks:
                m = block_masks[b_idx]
                redundant = True
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    if coverage[idx] == 1:
                        redundant = False
                        break
                    mm ^= lsb
                
                if redundant:
                    removed_any = True
                    # Don't add to new list
                else:
                    new_current_blocks.append(b_idx)
            
            current_blocks = new_current_blocks
            if not removed_any:
                break
        
        if len(current_blocks) < best_num_blocks:
            best_num_blocks = len(current_blocks)
            best_blocks = current_blocks
            
        # If we hit a very low number, we might stop early, but let's run a few more just in case
        # Actually, if we find 151, it's likely optimal or near optimal.
        if best_num_blocks <= 151:
             # We can break early if we are confident, but let's finish the restart or break if time is tight
             pass

    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback: return empty or single block? No, must be valid.
        # If no blocks found (shouldn't happen), return a valid covering (e.g., all blocks)
        # But greedy always finds one.
        return [list(k_blocks[i]) for i in range(num_k_blocks)]
        
    return [list(k_blocks[i]) for i in best_blocks]
