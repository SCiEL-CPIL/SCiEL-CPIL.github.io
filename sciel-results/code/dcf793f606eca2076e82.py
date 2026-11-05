<<<<<<< SEARCH
import itertools
import time

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    # Map each t-subset to a unique integer index
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    # A block is a k-subset. It covers all t-subsets contained within it.
    # We represent the set of covered t-subsets as a bitmask (Python int).
    # Bit i is 1 if t-subset i is covered.
    
    # Note: n_t_subs is 3876. Python ints can handle this easily.
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is the bitmask of t-subsets covered by block i
    block_masks = [0] * n_k_blocks
    
    # Map from block tuple to its index in all_k_blocks for quick lookup if needed
    # But we just need the list of blocks and their masks.
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        # Iterate over all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Construction
    uncovered_mask = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    # To speed up greedy, we can sort blocks by initial coverage (all same) 
    # or just iterate.
    # We will iterate through all blocks to find the best one each time.
    
    while uncovered_mask != 0:
        if time.time() > deadline - 20:
            break
            
        best_idx = -1
        best_gain = -1
        
        # Optimization: Skip blocks that are already selected?
        # No, we need to find the best *new* block.
        # We can maintain a list of active block indices or just iterate all.
        # Iterating all 50k is fast enough.
        
        for i in range(n_k_blocks):
            # Calculate gain: how many currently uncovered subsets does this block cover?
            # gain = popcount(block_masks[i] & uncovered_mask)
            gain = (block_masks[i] & uncovered_mask).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early exit if we found a perfect match (covers all remaining)? Unlikely.
                # If gain == 0, we can break if we've checked some, but best_gain starts -1.
        
        if best_idx == -1 or best_gain == 0:
            # Should not happen if uncovered_mask != 0
            break
            
        selected_block_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 4. Convert indices back to blocks
    blocks = [list(all_k_blocks[i]) for i in selected_block_indices]
    
    # 5. Local Search: Redundancy Removal
    # We want to remove any block that is not essential.
    # A block is essential if it is the *only* block covering at least one t-subset.
    
    # Precompute cover counts for each t-subset
    cover_count = [0] * n_t_subs
    
    # For each selected block, increment count for its t-subsets
    selected_masks = [block_masks[i] for i in selected_block_indices]
    
    for mask in selected_masks:
        # Iterate over set bits in mask
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    # Pass 1: Remove non-critical blocks
    # We iterate through blocks and check if any t-subset in it has cover_count == 1.
    # If not, the block can be removed.
    
    # To make this efficient, we can maintain the cover_count and update it.
    # However, removing a block changes cover_counts for all other blocks' subsets.
    # A simple approach:
    # 1. Build a list of (block_idx, mask).
    # 2. Iterate. For each block, check if it's critical.
    # 3. If not, remove it and decrement cover_counts for its subsets.
    # 4. Repeat until no more removals.
    
    # Since the number of blocks is small (~150), we can afford to be slightly inefficient.
    
    current_blocks = selected_block_indices[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    # Re-initialize cover_count for the current set
    cover_count = [0] * n_t_subs
    for mask in current_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
        for i in range(len(current_blocks)):
            mask = current_masks[i]
            # Check if this block is the only cover for any subset
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
                
            if is_critical:
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                # Remove this block
                # Decrement cover counts
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
        
    # Final blocks
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
=======
import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    rng = random.Random(42)
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is the bitmask of t-subsets covered by block i
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    full_mask = (1 << n_t_subs) - 1
    
    def greedy_construct(alpha=1.0, seed=0):
        """Randomized greedy: at each step, pick among top-gain blocks with probability proportional to gain^alpha"""
        local_rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered != 0:
            if time.time() > deadline - 15:
                break
                
            # Find all gains
            gains = [0] * n_k_blocks
            max_gain = 0
            for i in range(n_k_blocks):
                g = (block_masks[i] & uncovered).bit_count()
                gains[i] = g
                if g > max_gain:
                    max_gain = g
            
            if max_gain == 0:
                break
                
            # Weighted random selection: weight = gain^alpha
            # For alpha=0, uniform over all blocks with gain>0
            # For alpha=1, proportional to gain
            # For large alpha, close to greedy
            
            # Collect candidate blocks (gain > 0)
            candidates = []
            weights = []
            for i in range(n_k_blocks):
                if gains[i] > 0:
                    candidates.append(i)
                    if alpha == 0:
                        weights.append(1.0)
                    elif alpha == 1:
                        weights.append(float(gains[i]))
                    else:
                        weights.append(float(gains[i]) ** alpha)
            
            # Weighted random pick
            total_w = sum(weights)
            r = local_rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for j, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = candidates[j]
                    break
            
            selected.append(chosen)
            uncovered &= ~block_masks[chosen]
        
        return selected
    
    def redundancy_removal(block_indices):
        """Remove blocks that are not the sole cover of any t-subset"""
        if not block_indices:
            return []
        
        # Build cover counts
        cover_count = [0] * n_t_subs
        masks = [block_masks[i] for i in block_indices]
        
        for mask in masks:
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        # Iteratively remove non-critical blocks
        current = list(block_indices)
        current_masks = masks[:]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_current = []
            new_current_masks = []
            
            for i in range(len(current)):
                mask = current_masks[i]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if is_critical:
                    new_current.append(current[i])
                    new_current_masks.append(mask)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    improved = True
            
            current = new_current
            current_masks = new_current_masks
        
        return current
    
    def swap_improve(block_indices, max_iters=50):
        """Try to swap one block for another to reduce total block count"""
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        n = len(current)
        
        # Build cover counts
        cover_count = [0] * n_t_subs
        for idx in current:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
        
        # Find critical subsets (cover_count == 1)
        # For each block, check if it can be swapped with a better one
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5 and iterations < max_iters:
            improved = False
            iterations += 1
            
            for pos in range(len(current)):
                if time.time() > deadline - 5:
                    break
                    
                block_to_remove = current[pos]
                mask_remove = block_masks[block_to_remove]
                
                # Check if this block is critical
                is_critical = False
                m = mask_remove
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if is_critical:
                    continue
                
                # Try to find a replacement that covers the same critical subsets
                # Actually, if not critical, we can just remove it
                # But swap_improve is for when we want to keep the same size but maybe improve
                # For now, just do removal
                m = mask_remove
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    cover_count[i] -= 1
                    m ^= lsb
                
                current.pop(pos)
                improved = True
                break
        
        return current
    
    # 3. Multiple randomized greedy passes with different alphas
    best_blocks = None
    best_count = float('inf')
    
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    seeds = range(20)
    
    for alpha in alphas:
        for seed in seeds:
            if time.time() > deadline - 10:
                break
                
            selected = greedy_construct(alpha=alpha, seed=seed)
            if not selected:
                continue
            
            # Redundancy removal
            pruned = redundancy_removal(selected)
            
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_blocks = pruned[:]
    
    # 4. Final local search on the best solution
    if best_blocks:
        best_blocks = swap_improve(best_blocks)
        # Another round of redundancy removal after swaps
        best_blocks = redundancy_removal(best_blocks)
    
    # 5. Convert to list of lists
    blocks = [list(all_k_blocks[i]) for i in best_blocks] if best_blocks else []
    
    return blocks
>>>>>>> REPLACE
