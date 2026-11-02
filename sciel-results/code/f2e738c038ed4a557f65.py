import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    # Pre-compute masks for all k-blocks
    # mask[i] is a bitmask of t-subsets covered by block i
    block_masks = [0] * n_blocks
    # Also pre-compute for each t-subset, which blocks cover it
    # t_sub_to_blocks[j] is a list of block indices that cover t-subset j
    t_sub_to_blocks = [[] for _ in range(n_subsets)]
    
    # This loop is heavy but done once. 
    # C(24,7) = 346,104 blocks.
    # For each block, iterate C(7,3) = 35 t-subsets.
    # Total operations ~ 12 million.
    
    # Optimization: Generate t-subsets of each block and update reverse index
    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    # Check time
    if time.time() > deadline:
        # Fallback if precomputation takes too long (unlikely for v=24)
        return greedy_fallback(v, k, t, deadline)

    best_blocks = None
    best_score = float('inf')
    
    # Helper to count bits efficiently
    def bit_count(x):
        return x.bit_count()
        
    # Greedy construction with pre-computed data
    for run in range(50): # Cap runs
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to base the block on
            # Find a set bit in uncovered_mask
            # Using random sampling might be faster than finding lowest bit if density is low
            # But initially density is high.
            # Let's pick a random index and check if covered. If not, use it.
            # If many tries fail, find lowest set bit.
            target_idx = -1
            # Try 10 random picks
            for _ in range(10):
                pos = random.randint(0, n_subsets - 1)
                if (uncovered_mask >> pos) & 1:
                    target_idx = pos
                    break
            
            if target_idx == -1:
                # Find lowest set bit
                temp = uncovered_mask
                target_idx = (temp & -temp).bit_length() - 1
            
            # Get list of blocks covering this t-subset
            candidates = t_sub_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            # Iterate through candidates
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen if valid covering exists
                break
            
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            
            # Update uncovered mask
            uncovered_mask &= ~block_masks[best_block_idx]
        
        if not uncovered_mask: # Valid covering found
            # Post-processing: Remove redundant blocks
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    # Check if b covers any subset uniquely
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove b
                        changed = True
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                
                blocks = new_blocks
                
                if time.time() > deadline:
                    break
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()
                
    if best_blocks is None:
        # Fallback: deterministic greedy using precomputed data
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        while uncovered_mask:
            temp = uncovered_mask
            target_idx = (temp & -temp).bit_length() - 1
            candidates = t_sub_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        best_blocks = blocks

    return best_blocks

def greedy_fallback(v, k, t, deadline):
    # Simple fallback if precomputation fails or times out
    start_time = time.time()
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    def block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        return mask
            
    uncovered_mask = (1 << n_subsets) - 1
    blocks = []
    while uncovered_mask:
        if time.time() > deadline:
            break
        temp = uncovered_mask
        target_idx = (temp & -temp).bit_length() - 1
        target = all_t_subsets[target_idx]
        
        remaining_points = [p for p in range(v) if p not in target]
        best_block = None
        best_gain = -1
        for combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(target + combo))
            cand_mask = block_mask(candidate_block)
            gain = bin(uncovered_mask & cand_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
        blocks.append(list(best_block))
        best_mask = block_mask(best_block)
        uncovered_mask &= ~best_mask
    return blocks
