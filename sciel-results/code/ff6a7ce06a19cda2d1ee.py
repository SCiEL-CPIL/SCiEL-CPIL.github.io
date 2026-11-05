import itertools
import sys
import random
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    block_masks = [0] * n_k_blocks
    # Also precompute the list of t-subset indices for each block for faster iteration
    block_t_indices = [None] * n_k_blocks
    
    for i, block in enumerate(k_blocks):
        mask = 0
        t_indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            t_indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = t_indices

    # 3. Greedy Set Cover for initialization
    uncovered = (1 << n_t_subs) - 1
    selected_indices = set()
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            if i in selected_indices:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected_indices.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search with Dynamic Clause Weighting (LSAP)
    # We want to minimize |selected_indices| while keeping uncovered == 0.
    # Penalty = |selected_indices| + Lambda * |uncovered|
    # We use dynamic weights for uncovered subsets to guide "repair" steps.
    
    current_solution = list(selected_indices)
    best_solution = current_solution[:]
    best_size = len(current_solution)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    # Helper to remove redundant blocks
    def remove_redundant(blocks):
        blocks = list(blocks)
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        counts = [0] * n_t_subs
        for b in blocks:
            for idx in block_t_indices[b]:
                counts[idx] += 1
                
        new_blocks = []
        for b in blocks:
            # Check if block b has any unique coverage
            essential = False
            for idx in block_t_indices[b]:
                if counts[idx] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(b)
            else:
                # Decrement counts for this block
                for idx in block_t_indices[b]:
                    counts[idx] -= 1
        return new_blocks

    # Main Local Search Loop
    # We will perform restarts.
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with best solution and perturb
        current = best_solution[:]
        
        # Perturbation: Remove a few random blocks to allow repair to find different structure
        # Number of removals depends on restart
        num_removals = min(3 + restart // 5, len(current) // 2)
        random.shuffle(current)
        for _ in range(num_removals):
            if current:
                current.pop()
                
        # Phase 1: Remove redundant blocks (might be invalid now, so just cleanup if valid)
        # If invalid, we skip redundancy removal until valid
        covered_mask = 0
        for b in current:
            covered_mask |= block_masks[b]
        is_valid = (covered_mask == (1 << n_t_subs) - 1)
        
        if is_valid:
            current = remove_redundant(current)
            
        # Phase 2: Repair and Optimize Loop
        # Weights for dynamic clause weighting
        weights = [1.0] * n_t_subs
        
        local_time_limit = 2.0
        local_start = time.time()
        
        while time.time() - local_start < local_time_limit:
            # Recalculate state
            covered_mask = 0
            for b in current:
                covered_mask |= block_masks[b]
            uncovered_mask = ((1 << n_t_subs) - 1) ^ covered_mask
            
            if uncovered_mask == 0:
                # Valid solution
                # Try to remove redundant blocks
                new_current = remove_redundant(current)
                if len(new_current) < len(current):
                    current = new_current
                    # Update best
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = current[:]
                        # Decay weights
                        for i in range(n_t_subs):
                            weights[i] *= 0.95
                else:
                    # No redundancy, try to perturb to escape local minimum
                    # Remove a random block (making it invalid)
                    if current:
                        # Pick a block that covers many subsets with high redundancy
                        # Simple heuristic: Random
                        idx_to_remove = random.randint(0, len(current) - 1)
                        current.pop(idx_to_remove)
                        # Do not update weights here, let repair step handle it
                    else:
                        break
            else:
                # Invalid: Need to add blocks
                # Update weights: boost uncovered subsets
                m = uncovered_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    weights[idx] *= 1.05
                    m ^= lsb
                # Decay covered weights slightly to prevent explosion
                # (Optional, might skip for speed)
                
                # Find best block to add
                best_idx = -1
                best_score = -1.0
                # Only consider blocks that cover at least one uncovered subset
                for i in range(n_k_blocks):
                    if i in current:
                        continue
                    mask_i = block_masks[i]
                    common = mask_i & uncovered_mask
                    if common == 0:
                        continue
                    # Score: sum of weights of subsets covered by i that are currently uncovered
                    score = 0.0
                    mc = common
                    while mc:
                        lsb = mc & -mc
                        idx = lsb.bit_length() - 1
                        score += weights[idx]
                        mc ^= lsb
                    
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        
                if best_idx != -1:
                    current.append(best_idx)
                else:
                    break # Should not happen

    # Final cleanup: remove redundancies one last time
    final_solution = remove_redundant(best_solution)
    
    return [list(k_blocks[i]) for i in final_solution]
