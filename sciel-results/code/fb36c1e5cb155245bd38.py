import itertools
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # Precompute all k-subsets and their coverage masks
    # mask is an integer where bit i is 1 if the k-subset covers t-sub i
    k_subs = list(itertools.combinations(range(v), k))
    k_sub_masks = []
    for k_sub in k_subs:
        mask = 0
        # Generate all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        k_sub_masks.append(mask)
        
    # Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    blocks = []
    used_masks = []
    
    # To speed up max search, we can just iterate. 75k iterations per block is fine.
    # Number of blocks ~ 300-400. 300 * 75k = 22.5M operations. Might be slow in Python.
    # Optimization: We can skip blocks that are already used? No, blocks are distinct.
    # We can skip blocks that have 0 overlap.
    
    # A faster approach for max overlap:
    # We can maintain a list of candidate blocks.
    # But simple iteration might be okay if optimized.
    # Let's try simple iteration first. If too slow, we can use a heap or other structures.
    
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    while uncovered:
        best_mask = 0
        best_idx = -1
        best_count = 0
        
        # Find block with max new coverage
        for i, mask in enumerate(k_sub_masks):
            # Quick check: if mask & uncovered == 0, skip
            # But we need to find the max.
            # We can prune if (mask & uncovered).bit_count() <= best_count
            # But bit_count is expensive.
            
            # Let's just compute it.
            new_cov = mask & uncovered
            if new_cov:
                cnt = new_cov.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_mask = mask
                    best_idx = i
        
        if best_idx == -1:
            # Should not happen if covering is possible
            break
            
        blocks.append(k_subs[best_idx])
        used_masks.append(best_mask)
        uncovered &= ~best_mask
        
        if time.time() > deadline:
            break
            
    if not uncovered:
        # Reduction phase
        # Count coverage for each t-subset
        cover_count = [0] * n_t_subs
        
        # Build cover counts
        for mask in used_masks:
            # Iterate set bits
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_used_masks = []
            for i, mask in enumerate(used_masks):
                # Try removing this block
                # Decrement counts
                m = mask
                zero_found = False
                affected_idxs = []
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    affected_idxs.append(idx)
                    if cover_count[idx] == 0:
                        zero_found = True
                        break
                    m ^= lsb
                
                if zero_found:
                    # Restore counts
                    for idx in affected_idxs:
                        cover_count[idx] += 1
                    # Keep block
                    new_blocks.append(blocks[i])
                    new_used_masks.append(mask)
                else:
                    # Remove block, counts stay decremented
                    changed = True
                    # No need to restore
            
            blocks = new_blocks
            used_masks = new_used_masks
            
            if time.time() > deadline:
                break

    return blocks
