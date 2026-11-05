import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to bitmask of t-subsets it covers
    # Using Python integers as bitsets
    block_masks = [0] * n_blocks
    # Also store the list of t-subset indices for faster iteration in some cases?
    # Bitmask popcount is fast in Python 3.11+? int.bit_count()
    
    for b_idx, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_index[sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << n_t_subs) - 1
    
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, mask in enumerate(block_masks):
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_to_blocks[idx].append(b_idx)
            m ^= lsb

    # Helper: Greedy construction
    def greedy_init():
        uncovered = full_mask
        solution_set = set()
        solution_list = []
        
        # To speed up greedy, we can just scan all blocks each time?
        # 50k blocks * 150 iterations = 7.5M operations. Feasible.
        # But we can be smarter.
        
        while uncovered:
            if time.time() > deadline:
                break
            best_b_idx = -1
            max_gain = 0
            
            # Optimization: Only consider blocks that cover at least one uncovered subset
            # Pick a random uncovered subset to filter candidates?
            # Or just scan all.
            # Let's scan all.
            for b_idx in range(n_blocks):
                if b_idx in solution_set:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_b_idx = b_idx
                    if max_gain == 0:
                        break # Should not happen if uncovered != 0 and valid problem
            
            if best_b_idx == -1:
                break # Should not happen
            
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            uncovered &= ~block_masks[best_b_idx]
            
        return solution_list, solution_set

    def calculate_cover_counts(solution_list):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        return cover_count

    def local_search(solution_list, solution_set):
        cover_count = calculate_cover_counts(solution_list)
        improved = True
        
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Pass 1: Deletion
            # Randomize order to avoid bias
            indices = list(range(len(solution_list)))
            random.shuffle(indices)
            
            for i in indices:
                b_idx = solution_list[i]
                # Check if redundant
                redundant = True
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove block
                    m = block_masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break # Restart deletion pass
                    
            # Pass 2: Swap
            if not improved:
                indices = list(range(len(solution_list)))
                random.shuffle(indices)
                
                for i in indices:
                    b_idx = solution_list[i]
                    # Find critical subsets
                    critical = []
                    m = block_masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical.append(idx)
                        m ^= lsb
                        
                    if not critical:
                        # Redundant, should have been caught by deletion
                        # But maybe deletion pass missed it due to order?
                        # Just delete it.
                        m = block_masks[b_idx]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find candidates
                    # Intersect tsub_to_blocks for all critical subsets
                    if len(critical) == 1:
                        candidates = tsub_to_blocks[critical[0]]
                    else:
                        # Start with smallest list
                        min_idx = min(critical, key=lambda x: len(tsub_to_blocks[x]))
                        candidates = set(tsub_to_blocks[min_idx])
                        for s in critical:
                            if s == min_idx:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                    
                    if not candidates:
                        continue
                        
                    # Filter out current solution
                    candidates = [c for c in candidates if c not in solution_set]
                    if not candidates:
                        continue
                        
                    # Choose best candidate
                    # Heuristic: Maximize number of subsets covered by candidate that are currently covered by exactly 2 blocks
                    # This increases redundancy, helping future deletions.
                    best_candidate = -1
                    best_score = -1
                    
                    for c in candidates:
                        score = 0
                        m = block_masks[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            if cover_count[idx] == 2:
                                score += 1
                            m ^= lsb
                            
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                            
                    if best_candidate != -1:
                        # Perform Swap
                        # Decrement cover_count for b_idx
                        m = block_masks[b_idx]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                            
                        # Increment cover_count for best_candidate
                        m = block_masks[best_candidate]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                            
                        # Update solution
                        solution_list[i] = best_candidate
                        solution_set.remove(b_idx)
                        solution_set.add(best_candidate)
                        improved = True
                        break
                        
        return solution_list, solution_set

    # Helper: Perturbation and Repair
    def perturb_and_repair(solution_list, solution_set):
        n_remove = random.randint(3, 6)
        if len(solution_list) <= n_remove:
            return greedy_init()
            
        indices_to_remove = random.sample(range(len(solution_list)), n_remove)
        indices_to_remove.sort(reverse=True)
        
        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)
            
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove]
        
        # Calculate uncovered
        covered_mask = 0
        for b_idx in new_solution_list:
            covered_mask |= block_masks[b_idx]
            
        uncovered_mask = full_mask & ~covered_mask
        
        if uncovered_mask == 0:
            return new_solution_list, new_solution_set
            
        # Greedy Repair
        # Add blocks to cover uncovered_mask
        while uncovered_mask:
            if time.time() > deadline:
                break
            best_b_idx = -1
            max_gain = 0
            
            # Find block with max gain on uncovered_mask
            # Optimization: Only consider blocks covering at least one uncovered bit
            # Pick a random uncovered bit to find candidates
            # Or just scan all.
            
            # To speed up, pick a random uncovered subset index
            # Get list of blocks covering it
            m = uncovered_mask
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates = tsub_to_blocks[idx]
            
            for b_idx in candidates:
                if b_idx in new_solution_set:
                    continue
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_b_idx = b_idx
                    if max_gain == (uncovered_mask.bit_count()): # Covered all? Unlikely
                        break
            
            if best_b_idx != -1:
                new_solution_set.add(best_b_idx)
                new_solution_list.append(best_b_idx)
                uncovered_mask &= ~block_masks[best_b_idx]
            else:
                # Fallback: scan all blocks
                for b_idx in range(n_blocks):
                    if b_idx in new_solution_set:
                        continue
                    gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_b_idx = b_idx
                if best_b_idx != -1:
                    new_solution_set.add(best_b_idx)
                    new_solution_list.append(best_b_idx)
                    uncovered_mask &= ~block_masks[best_b_idx]
                else:
                    break
                    
        return new_solution_list, new_solution_set

    # Main ILS Loop
    random.seed(42)
    
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set
    
    while time.time() < deadline - 5.0:
        # Perturb
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)
        
        # Local Search on perturbed
        opt_list, opt_set = local_search(pert_list, pert_set)
        
        # Update
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
        else:
            # Accept if better than current
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set
            # Sometimes accept equal or slightly worse to escape local minima?
            # Standard ILS usually accepts only improvements or uses SA.
            # Let's stick to improvements for stability, but maybe add a random restart if stuck?
            # If we are stuck, we might need to jump.
            # For now, simple hill climbing.
            
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
