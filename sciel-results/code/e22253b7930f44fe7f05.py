import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Map block index to bitmask of t-subsets it covers
    block_masks = [0] * n_k_subsets
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            ts_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << n_t_subsets) - 1
    
    def popcount(x):
        return x.bit_count()
        
    def greedy_construct(randomize=False):
        covered_mask = 0
        selected_blocks = []
        coverage_counts = [0] * n_t_subsets
        uncovered_indices = list(range(n_t_subsets))
        
        while uncovered_indices:
            if randomize:
                idx_pos = random.randint(0, len(uncovered_indices) - 1)
                target_idx = uncovered_indices[idx_pos]
            else:
                idx_pos = random.randint(0, len(uncovered_indices) - 1)
                target_idx = uncovered_indices[idx_pos]
            
            best_block = -1
            best_gain = -1
            candidates = ts_to_blocks[target_idx]
            
            for b_idx in candidates:
                new_mask = block_masks[b_idx] & ~covered_mask
                gain = popcount(new_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            selected_blocks.append(best_block)
            new_mask = block_masks[best_block] & ~covered_mask
            covered_mask |= block_masks[best_block]
            
            m = new_mask
            while m:
                idx = (m & -m).bit_length() - 1
                coverage_counts[idx] += 1
                m &= m - 1
            
            uncovered_indices = [i for i in range(n_t_subsets) if coverage_counts[i] == 0]
            
        return selected_blocks

    def local_search(initial_blocks):
        if not initial_blocks:
            return []
            
        current_blocks = list(initial_blocks)
        coverage_counts = [0] * n_t_subsets
        unique_mask = 0
        covered_mask = 0
        
        for b_idx in current_blocks:
            m = block_masks[b_idx]
            covered_mask |= m
            mm = m
            while mm:
                idx = (mm & -mm).bit_length() - 1
                coverage_counts[idx] += 1
                if coverage_counts[idx] == 1:
                    unique_mask |= (1 << idx)
                mm &= mm - 1
                
        # Repair Phase: If not fully covered, add blocks
        if covered_mask != full_mask:
            while covered_mask != full_mask:
                # Find an uncovered subset
                # We can just find the first one in uncovered_indices
                # But we don't have uncovered_indices here.
                # Let's find one by checking bits in ~covered_mask
                missing_mask = ~covered_mask & full_mask
                if missing_mask == 0:
                    break
                idx = (missing_mask & -missing_mask).bit_length() - 1
                
                best_block = -1
                best_gain = -1
                candidates = ts_to_blocks[idx]
                
                for b_idx in candidates:
                    if b_idx in current_blocks:
                        continue
                    new_mask = block_masks[b_idx] & ~covered_mask
                    gain = popcount(new_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        
                if best_block == -1:
                    break
                    
                current_blocks.append(best_block)
                m = block_masks[best_block]
                covered_mask |= m
                mm = m
                while mm:
                    i = (mm & -mm).bit_length() - 1
                    coverage_counts[i] += 1
                    if coverage_counts[i] == 1:
                        unique_mask |= (1 << i)
                    mm &= mm - 1

        # Optimization Phase
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            
            # Deletion Pass
            to_remove = -1
            for i, b_idx in enumerate(current_blocks):
                if (block_masks[b_idx] & unique_mask) == 0:
                    to_remove = i
                    break
                    
            if to_remove != -1:
                b_idx = current_blocks[to_remove]
                m = block_masks[b_idx]
                mm = m
                while mm:
                    idx = (mm & -mm).bit_length() - 1
                    coverage_counts[idx] -= 1
                    if coverage_counts[idx] == 1:
                        unique_mask |= (1 << idx)
                    mm &= mm - 1
                current_blocks.pop(to_remove)
                improved = True
                continue
                
            # Swap Pass
            candidates_for_swap = [b for b in current_blocks if (block_masks[b] & unique_mask) != 0]
            if not candidates_for_swap:
                break
                
            old_b = random.choice(candidates_for_swap)
            unique_subsets_mask = block_masks[old_b] & unique_mask
            
            idx_one = (unique_subsets_mask & -unique_subsets_mask).bit_length() - 1
            candidates = ts_to_blocks[idx_one]
            
            valid_new_blocks = []
            for nb in candidates:
                if nb == old_b:
                    continue
                if (block_masks[nb] & unique_subsets_mask) == unique_subsets_mask:
                    valid_new_blocks.append(nb)
                    
            if not valid_new_blocks:
                continue
                
            new_b = random.choice(valid_new_blocks)
            
            # Swap
            m_old = block_masks[old_b]
            mm = m_old
            while mm:
                idx = (mm & -mm).bit_length() - 1
                coverage_counts[idx] -= 1
                if coverage_counts[idx] == 1:
                    unique_mask |= (1 << idx)
                mm &= mm - 1
                
            m_new = block_masks[new_b]
            mm = m_new
            while mm:
                idx = (mm & -mm).bit_length() - 1
                coverage_counts[idx] += 1
                if coverage_counts[idx] == 2:
                    unique_mask &= ~(1 << idx)
                mm &= mm - 1
                
            current_blocks.remove(old_b)
            current_blocks.append(new_b)
            improved = True
            
        return current_blocks

    pop_size = 20
    population = []
    
    for i in range(pop_size):
        if time.time() - start_time > time_limit * 0.2:
            break
        blocks = greedy_construct(randomize=True)
        optimized = local_search(blocks)
        population.append(optimized)
        
    if not population:
        blocks = greedy_construct(randomize=False)
        population = [local_search(blocks)]

    best_solution = min(population, key=len)
    
    iterations = 0
    while time.time() - start_time < time_limit * 0.9:
        iterations += 1
        
        population.sort(key=len)
        parent = population[0]
        child = list(parent)
        
        if len(child) > 0:
            idx = random.randint(0, len(child) - 1)
            new_block = random.randint(0, n_k_subsets - 1)
            child[idx] = new_block
        
        optimized_child = local_search(child)
        population.append(optimized_child)
        
        population.sort(key=len)
        population = population[:pop_size]
        
        if len(population[0]) < len(best_solution):
            best_solution = population[0]

    final_blocks = local_search(best_solution)
    
    return [list(all_k_subsets[i]) for i in final_blocks]
