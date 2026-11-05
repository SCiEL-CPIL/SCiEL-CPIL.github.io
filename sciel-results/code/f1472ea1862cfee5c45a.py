import itertools
import sys
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    # Precompute mask for each block
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search: Deletion and Swap
    # Maintain counts of how many selected blocks cover each t-subset
    counts = [0] * n_t_subs
    selected_set = set(selected_block_indices)
    
    # Initialize counts
    for idx in selected_block_indices:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
            
    # Helper to update counts
    def update_counts(idx, delta):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += delta
            m ^= lsb

    improved = True
    # Limit iterations to prevent infinite loops or excessive time
    max_iterations = 50 
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Pass 1: Deletion
        # Try to remove any block that is not essential
        to_remove = []
        for idx in selected_block_indices:
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                to_remove.append(idx)
                
        for idx in to_remove:
            selected_block_indices.remove(idx)
            selected_set.remove(idx)
            update_counts(idx, -1)
            improved = True
            
        # Pass 2: Swap
        # For each selected block A, find if there is an unselected block B
        # that covers all unique subsets of A.
        # If so, swap A with B. This might enable further deletions.
        
        # To make this efficient, we only consider swaps if A has unique subsets.
        # We iterate through a copy of selected_block_indices because we modify the set.
        # Note: Swapping one block might make another block removable, which is handled in the next Deletion Pass.
        
        # Optimization: Only try swaps for blocks with small unique coverage?
        # Actually, any swap that maintains coverage is valid.
        
        # We need to find B such that mask[B] covers unique_mask[A].
        # unique_mask[A] = mask[A] & (bits where count==1)
        
        # To avoid O(N^2) slow loops in Python, we can be selective.
        # If the number of selected blocks is large, this might be slow.
        # However, N=75k, S=335.
        # We can limit the number of swap attempts per iteration or per block.
        
        # Let's try a limited number of random swaps or a structured search.
        # Given the time limit, a full O(S*N) scan might be too slow if done many times.
        # 335 * 75000 = 25M operations. In Python, 25M simple ops takes ~1-2 seconds.
        # If we do this 50 times, it's 50-100 seconds. Risky.
        
        # Strategy: Only perform swap if we haven't improved in the last deletion pass?
        # Or, perform a limited number of swaps.
        
        # Let's do a single pass of swaps.
        # We will pick one swap per iteration if found, or scan a subset.
        
        # To be safe on time, we will only attempt swaps if the current size is > 335 (best known).
        # Or just attempt a few random swaps.
        
        # Actually, a deterministic "first found" swap is better for consistency.
        
        # We will iterate through selected blocks. For each, we check if it has unique subsets.
        # If yes, we look for a replacement.
        # To speed up replacement search, we can pre-compute for each t-subset which blocks cover it.
        
        # Pre-compute: sub_to_blocks[sub_idx] = list of block indices covering sub_idx
        # This is memory heavy: 11628 subsets * ~1000 blocks each?
        # Total entries = sum over blocks of C(k,t) = 75582 * 56 = 4.2M entries.
        # This is manageable.
        
        if not hasattr(construct, '_sub_to_blocks'):
            construct._sub_to_blocks = [[] for _ in range(n_t_subs)]
            for i, block in enumerate(k_blocks):
                for t_sub in itertools.combinations(block, t):
                    idx = sub_to_idx[t_sub]
                    construct._sub_to_blocks[idx].append(i)
        
        sub_to_blocks = construct._sub_to_blocks
        
        swap_done = False
        for A in list(selected_block_indices):
            if swap_done:
                break
                
            # Compute unique mask for A
            m = block_masks[A]
            unique_mask = 0
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_mask |= lsb
                m ^= lsb
                
            if unique_mask == 0:
                continue # Should have been removed in deletion pass
            
            # Find candidates B that cover unique_mask
            # Intersection of sub_to_blocks for all bits in unique_mask
            
            # Start with the list of blocks for the first bit in unique_mask
            lsb = unique_mask & -unique_mask
            first_bit = lsb.bit_length() - 1
            candidates = set(sub_to_blocks[first_bit])
            
            # Intersect with lists for other bits
            m = unique_mask
            m ^= lsb # skip first
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                # Intersect candidates with sub_to_blocks[bit_idx]
                # If the list is small, iterate it. If large, iterate candidates.
                # Usually, the intersection shrinks quickly.
                next_blocks = sub_to_blocks[bit_idx]
                # Optimization: if next_blocks is smaller, iterate it and check membership in candidates
                if len(next_blocks) < len(candidates):
                    new_candidates = set()
                    for b in next_blocks:
                        if b in candidates:
                            new_candidates.add(b)
                    candidates = new_candidates
                else:
                    candidates = set(b for b in candidates if b in next_blocks) # This is slow if candidates is large
                    # Better:
                    # candidates.intersection_update(next_blocks)
                    candidates.intersection_update(next_blocks)
                    
                if not candidates:
                    break
                m ^= lsb
                
            if candidates:
                # Choose a candidate B not in selected_set
                # Prefer one that covers many other subsets? Or just pick one.
                # To avoid swapping with a block that is already selected, filter.
                valid_candidates = [b for b in candidates if b not in selected_set]
                if valid_candidates:
                    B = valid_candidates[0] # Pick first
                    # Perform swap
                    selected_block_indices.remove(A)
                    selected_set.remove(A)
                    update_counts(A, -1)
                    
                    selected_block_indices.append(B)
                    selected_set.add(B)
                    update_counts(B, 1)
                    
                    improved = True
                    swap_done = True

    # Final Deletion Pass to clean up after swaps
    to_remove = []
    for idx in selected_block_indices:
        m = block_masks[idx]
        essential = False
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                essential = True
                break
            m ^= lsb
        if not essential:
            to_remove.append(idx)
            
    for idx in to_remove:
        selected_block_indices.remove(idx)
        selected_set.remove(idx)
        update_counts(idx, -1)

    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in selected_block_indices]
