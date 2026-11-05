"""Improved covering design C(v,k,t) using bitmask-based greedy with redundancy removal."""
import time
import itertools
import random

def construct(v, k, t):
    """Construct a covering of t-subsets by k-subsets of {0,...,v-1}."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Precompute all t-subsets as bitmasks
    # For v=19, k=8, t=5: C(19,5) = 11628 t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {}
    for i, sub in enumerate(all_t_subsets):
        t_subset_to_idx[sub] = i
    
    # Precompute bitmask for each t-subset
    t_subsets_mask = []
    for sub in all_t_subsets:
        mask = 0
        for x in sub:
            mask |= (1 << x)
        t_subsets_mask.append(mask)
    
    # Track which t-subsets are covered
    # Use a bytearray for efficiency: covered[i] = 1 if t-subset i is covered
    covered = bytearray(n_t)
    
    # For each k-subset (block), precompute which t-subsets it covers
    # C(19,8) = 75582 blocks, each covers C(8,5)=56 t-subsets
    # Precompute the list of t-subset indices covered by each possible block
    # But 75582 * 56 = ~4.2M entries, which is manageable
    
    # Actually, let's use a different approach: for each block, compute the set of t-subset indices it covers
    # We'll compute this on the fly but cache it
    
    # Better approach: for each point x, precompute which t-subsets contain x
    # Then for a block B, the t-subsets covered are those where all t points are in B
    
    # Let's use a smarter representation:
    # For each t-subset, store its bitmask
    # A block B covers a t-subset S if mask(S) is a subset of mask(B)
    
    # Greedy algorithm:
    # 1. Find the t-subset with the fewest covering blocks (or first uncovered)
    # 2. Create a block that covers it and maximizes new coverage
    # 3. Remove redundant blocks
    
    # Precompute: for each point, which t-subsets contain it
    point_to_tsubsets = [[] for _ in range(v)]
    for i, mask in enumerate(t_subsets_mask):
        for x in range(v):
            if mask & (1 << x):
                point_to_tsubsets[x].append(i)
    
    # Convert to sets for faster intersection
    point_to_tsubsets_set = [set(lst) for lst in point_to_tsubsets]
    
    # Greedy construction
    blocks = []
    block_masks = []  # bitmask for each block
    uncovered_count = n_t
    
    # Track coverage count for each t-subset
    cover_count = [0] * n_t
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset (first one for determinism, or one with min options)
        # For speed, just pick the first uncovered one
        target_idx = -1
        for i in range(n_t):
            if cover_count[i] == 0:
                target_idx = i
                break
        if target_idx == -1:
            break
        
        target_mask = t_subsets_mask[target_idx]
        target_points = [x for x in range(v) if target_mask & (1 << x)]
        
        # Build a block starting with the target points
        block = set(target_points)
        block_mask = target_mask
        
        # Add points one at a time to maximize new coverage
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute how many new t-subsets would be covered
                new_mask = block_mask | (1 << x)
                # Count t-subsets where all points are in new_mask but not all in block_mask
                # This is expensive to compute directly
                # Instead, estimate: for each t-subset containing x, check if other t-1 points are in block
                gain = 0
                for tidx in point_to_tsubsets_set[x]:
                    m = t_subsets_mask[tidx]
                    # Check if m is subset of new_mask
                    if m & ~new_mask == 0:
                        # And not subset of block_mask (i.e., x is needed)
                        if m & ~block_mask != 0:
                            if cover_count[tidx] == 0:
                                gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.add(best_point)
            block_mask |= (1 << best_point)
        
        if len(block) < k:
            # Fill with arbitrary points
            for x in range(v):
                if x not in block:
                    block.add(x)
                    block_mask |= (1 << x)
                if len(block) == k:
                    break
        
        # Add this block and update coverage
        block_list = sorted(block)
        blocks.append(block_list)
        block_masks.append(block_mask)
        
        # Update cover counts
        for i in range(n_t):
            m = t_subsets_mask[i]
            if m & ~block_mask == 0:
                if cover_count[i] == 0:
                    uncovered_count -= 1
                cover_count[i] += 1
        
        # Periodic redundancy removal
        if len(blocks) % 50 == 0 and time.time() < deadline - 5:
            # Remove redundant blocks
            new_cover_count = [0] * n_t
            kept_blocks = []
            kept_masks = []
            for bi, (b, bm) in enumerate(zip(blocks, block_masks)):
                # Check if this block is still needed
                needed = False
                for i in range(n_t):
                    m = t_subsets_mask[i]
                    if m & ~bm == 0 and cover_count[i] == 1:
                        needed = True
                        break
                if needed:
                    kept_blocks.append(b)
                    kept_masks.append(bm)
                    for i in range(n_t):
                        m = t_subsets_mask[i]
                        if m & ~bm == 0:
                            new_cover_count[i] += 1
            if len(kept_blocks) < len(blocks):
                blocks = kept_blocks
                block_masks = kept_masks
                cover_count = new_cover_count
                uncovered_count = sum(1 for c in cover_count if c == 0)
    
    # Final redundancy removal
    if time.time() < deadline - 2:
        new_cover_count = [0] * n_t
        kept_blocks = []
        kept_masks = []
        for bi, (b, bm) in enumerate(zip(blocks, block_masks)):
            needed = False
            for i in range(n_t):
                m = t_subsets_mask[i]
                if m & ~bm == 0 and cover_count[i] == 1:
                    needed = True
                    break
            if needed:
                kept_blocks.append(b)
                kept_masks.append(bm)
                for i in range(n_t):
                    m = t_subsets_mask[i]
                    if m & ~bm == 0:
                        new_cover_count[i] += 1
        blocks = kept_blocks
        cover_count = new_cover_count
    
    return blocks
