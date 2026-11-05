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
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Initial Greedy Solution
    uncovered = (1 << n_t_subs) - 1
    selected_indices = set()
    
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
            
        selected_indices.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    best_size = len(selected_indices)
    best_solution = list(selected_indices)
    current_solution = set(selected_indices)
    current_size = best_size
    
    # 4. Local Search (Min-SAT style)
    start_time = time.time()
    time_limit = 100.0
    
    # Helper: Greedy Repair
    def repair(uncovered_mask, selected_set):
        while uncovered_mask:
            best_idx = -1
            max_gain = 0
            # Optimization: If we find a block that covers all uncovered, break early
            target_gain = uncovered_mask.bit_count()
            
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
                    if max_gain == target_gain:
                        break
                        
            if best_idx == -1:
                break # Should not happen
                
            selected_set.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        return selected_set, uncovered_mask

    while time.time() - start_time < time_limit:
        if not current_solution:
            break
            
        # Pick a random block to remove
        block_to_remove = random.choice(list(current_solution))
        
        new_set = set(current_solution)
        new_set.discard(block_to_remove)
        
        # Calculate uncovered mask for new_set
        union_mask = 0
        for idx in new_set:
            union_mask |= block_masks[idx]
            
        new_uncovered = ((1 << n_t_subs) - 1) & ~union_mask
        
        if new_uncovered == 0:
            current_solution = new_set
            current_size = len(new_set)
            if current_size < best_size:
                best_size = current_size
                best_solution = list(current_solution)
            continue
        
        # Repair
        repaired_set, final_uncovered = repair(new_uncovered, new_set)
        
        if final_uncovered != 0:
            continue
            
        new_size = len(repaired_set)
        
        if new_size < best_size:
            best_size = new_size
            best_solution = list(repaired_set)
            current_solution = repaired_set
            current_size = new_size
        elif new_size == current_size:
            # Accept to continue search from this state
            current_solution = repaired_set
        # else: revert (do nothing)

    # 5. Final Polish: Remove redundant blocks from best_solution
    if best_solution:
        counts = [0] * n_t_subs
        for idx in best_solution:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        final_indices = []
        for idx in best_solution:
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
                
            if essential:
                final_indices.append(idx)
            else:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    m ^= lsb
                    
        best_solution = final_indices

    return [list(k_blocks[i]) for i in best_solution]
