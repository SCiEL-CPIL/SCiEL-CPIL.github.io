import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset tuple to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # block_mask[i] is an integer where bit j is 1 if t-subset j is covered by block i
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Full mask of all t-subsets
    full_mask = (1 << n_t_subsets) - 1
    
    # Greedy Set Cover
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = full_mask
        blocks = []
        block_indices = list(range(n_blocks))
        
        while uncovered:
            best_idx = -1
            best_gain = -1
            # Randomize tie-breaking or selection slightly to diversify
            # For speed, we check all blocks.
            # Optimization: If time is tight, we might skip some, but 75k is ok.
            
            # To speed up, we can shuffle the list of blocks? No, that's O(N log N) each step.
            # Just iterate.
            for i in block_indices:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if random.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered != 0
                
            blocks.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
            if time.time() > deadline:
                return None
                
        return blocks

    def prune_redundant(blocks):
        """Remove blocks that are not essential."""
        if not blocks:
            return blocks
            
        # Count how many times each t-subset is covered
        # Using a list of counts is faster than bitmask ops for counting?
        # n_t_subsets is 11628.
        counts = [0] * n_t_subsets
        
        # Calculate counts
        for b_idx in blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        # Identify essential blocks
        essential = [False] * len(blocks)
        for i, b_idx in enumerate(blocks):
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    essential[i] = True
                    break
                m ^= lsb
                
        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    def local_search_2for1(blocks):
        """
        Try to replace 2 blocks with 1 block.
        Returns a new list of blocks if an improvement is found, else None.
        """
        if len(blocks) < 2:
            return None
            
        # Create a set of block indices for fast lookup
        block_set = set(blocks)
        
        # Calculate cover mask
        cover_mask = 0
        for b_idx in blocks:
            cover_mask |= block_masks[b_idx]
            
        # We will try random pairs until we find a replacement or time runs out
        # To be efficient, we limit the number of attempts per call
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Pick two distinct blocks
            i, j = random.sample(range(len(blocks)), 2)
            b1 = blocks[i]
            b2 = blocks[j]
            
            # Calculate unique subsets covered by b1 or b2
            # U = (Mask(b1) | Mask(b2)) & ~(CoverMask - (Mask(b1) | Mask(b2)))
            # U = (Mask(b1) | Mask(b2)) & ~(CoverMask & ~(Mask(b1) | Mask(b2)))
            mask12 = block_masks[b1] | block_masks[b2]
            # Subsets covered by others
            others_mask = cover_mask & ~mask12
            # Unique to b1/b2
            unique_mask = mask12 & ~others_mask
            
            # We need to find a block b3 such that unique_mask is subset of block_masks[b3]
            # i.e., (unique_mask & ~block_masks[b3]) == 0
            # And b3 is not b1 or b2
            
            found = False
            best_b3 = -1
            # Scan all blocks
            # Optimization: We can skip blocks that are already in the cover?
            # Actually, if b3 is in the cover (and not b1, b2), it's a valid replacement.
            # If b3 is not in the cover, it's also valid.
            
            for b3 in range(n_blocks):
                if b3 == b1 or b3 == b2:
                    continue
                if (unique_mask & ~block_masks[b3]) == 0:
                    # Found a replacement
                    # Create new block list
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(b3)
                    return new_blocks
                    
            # If no replacement found for this pair, try another pair
            
        return None

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    # 1. Greedy
    # 2. Prune
    # 3. Local Search (2-for-1)
    # 4. Prune again
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover(seed=i)
        if blocks is None:
            continue
            
        blocks = prune_redundant(blocks)
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a few 2-for-1 moves
            for _ in range(50): # Limit inner loop
                if time.time() > deadline:
                    break
                new_blocks = local_search_2for1(blocks)
                if new_blocks is not None:
                    blocks = new_blocks
                    improved = True
                    # Prune after a successful move
                    blocks = prune_redundant(blocks)
                    break # Restart the inner loop with the new state
                    
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 335:
            break

    if best_blocks is None:
        # Fallback
        blocks = greedy_cover(seed=None)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []
        
    return [list(all_blocks[i]) for i in best_blocks]
