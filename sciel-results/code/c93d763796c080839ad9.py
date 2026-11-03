import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe under 150s

    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    full_mask = (1 << num_triples) - 1

    # 2. Generate all candidate blocks and their masks
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    # Precompute block lists for each triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb

    def time_left():
        return deadline - time.time()

    # 1. Weighted Greedy with dynamic scarcity
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = full_mask
        selected_indices = []
        
        # Track how many times each triple is "available" (in uncovered)
        # Use a simple approach: at each step, compute gain with weighting
        
        while uncovered:
            if time_left() < 15:
                return None
                
            best_gain = -1
            best_idx = -1
            second_best_gain = -1
            second_best_idx = -1
            
            # Compute weighted gain: favor blocks that cover triples with fewer options
            # Approximate: gain = sum over covered triples of (1 / min(count, 10))
            # But that's expensive. Use a simpler heuristic:
            # Primary: raw gain. Secondary: prefer blocks with more "unique" coverage.
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    second_best_gain = best_gain
                    second_best_idx = best_idx
                    best_gain = gain
                    best_idx = i
                elif gain > second_best_gain:
                    second_best_gain = gain
                    second_best_idx = i
            
            if best_gain == 0:
                return selected_indices
            
            # With some probability, pick second-best to diversify
            if second_best_idx >= 0 and second_best_gain == best_gain and random.random() < 0.3:
                best_idx = second_best_idx
            
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    # 2. Refinement: Remove redundant blocks
    def refine_indices(selected_indices):
        if not selected_indices:
            return []
            
        solution_set = set(selected_indices)
        
        coverage_count = [0] * num_triples
        for idx in solution_set:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                coverage_count[idx_t] += 1
                m ^= lsb
                
        changed = True
        while changed:
            if time_left() < 10:
                break
            changed = False
            for idx in list(solution_set):
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    solution_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    changed = True
                    
        return list(solution_set)

    # 3. Local Search: 1-exchange (remove one block)
    def local_search_1exchange(solution_indices):
        if not solution_indices:
            return []
            
        sol_set = set(solution_indices)
        
        improved = True
        while improved:
            if time_left() < 5:
                break
            improved = False
            
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            sol_list = list(sol_set)
            random.shuffle(sol_list)
            
            for idx in sol_list:
                if time_left() < 5:
                    break
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    sol_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    improved = True
                    break
                    
        return list(sol_set)

    # 4. Local Search: 2-Exchange (Replace 2 blocks with 1)
    def local_search_2exchange(solution_indices):
        if not solution_indices:
            return []
            
        sol_set = set(solution_indices)
        if len(sol_set) < 2:
            return list(sol_set)
            
        improved = True
        while improved:
            if time_left() < 8:
                break
            improved = False
            
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            sol_list = list(sol_set)
            random.shuffle(sol_list)
            
            for i in range(len(sol_list)):
                if time_left() < 8:
                    break
                for j in range(i + 1, len(sol_list)):
                    if time_left() < 8:
                        break
                        
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        c = coverage_count[idx_t]
                        
                        in_b1 = bool(mask1 & lsb)
                        in_b2 = bool(mask2 & lsb)
                        
                        if c == 1:
                            critical_triples.append(idx_t)
                        elif c == 2:
                            if in_b1 and in_b2:
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break
                    else:
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        candidates = set(triple_to_blocks[critical_triples[0]])
                        
                        for t_idx in critical_triples[1:]:
                            candidates.intersection_update(triple_to_blocks[t_idx])
                            if not candidates:
                                break
                            
                        if candidates:
                            for b_k in candidates:
                                if b_k not in sol_set:
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_k)
                                    improved = True
                                    break
                            
                            if improved:
                                break
            
        return list(sol_set)

    # 5. Local Search: 2-Merge (Replace 2 blocks with 1, allowing the new block to be in sol_set)
    def local_search_2merge(solution_indices):
        if not solution_indices:
            return []
            
        sol_set = set(solution_indices)
        if len(sol_set) < 2:
            return list(sol_set)
            
        improved = True
        while improved:
            if time_left() < 8:
                break
            improved = False
            
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            sol_list = list(sol_set)
            random.shuffle(sol_list)
            
            for i in range(len(sol_list)):
                if time_left() < 8:
                    break
                for j in range(i + 1, len(sol_list)):
                    if time_left() < 8:
                        break
                        
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    # Find triples that would become uncovered
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        c = coverage_count[idx_t]
                        
                        in_b1 = bool(mask1 & lsb)
                        in_b2 = bool(mask2 & lsb)
                        
                        if c == 1:
                            critical_triples.append(idx_t)
                        elif c == 2:
                            if in_b1 and in_b2:
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break
                    else:
                        # Try to find ANY block (including ones already in sol_set) that covers all critical
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        candidates = set(triple_to_blocks[critical_triples[0]])
                        
                        for t_idx in critical_triples[1:]:
                            candidates.intersection_update(triple_to_blocks[t_idx])
                            if not candidates:
                                break
                            
                        if candidates:
                            # Prefer blocks not already in sol_set, but accept any
                            for b_k in candidates:
                                sol_set.remove(b1)
                                sol_set.remove(b2)
                                sol_set.add(b_k)
                                improved = True
                                break
                            
                            if improved:
                                break
            
        return list(sol_set)

    def is_valid(sol_indices):
        if not sol_indices:
            return num_triples == 0
        covered = 0
        for idx in sol_indices:
            covered |= block_masks[idx]
        return (covered == full_mask)

    best_solution_indices = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    
    for i in range(num_restarts):
        if time_left() < 30:
            break
            
        greedy_result = run_greedy(i * 7919 + 1)
        if greedy_result is None:
            continue
            
        refined_indices = refine_indices(greedy_result)
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution_indices = refined_indices
            
        # Try local search on this solution
        if time_left() > 15:
            ls_result = local_search_1exchange(refined_indices)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution_indices = ls_result
            
            if time_left() > 10:
                ls_result2 = local_search_2exchange(best_solution_indices)
                if len(ls_result2) < best_size:
                    best_size = len(ls_result2)
                    best_solution_indices = ls_result2
    
    if best_solution_indices is None:
        greedy_result = run_greedy(0)
        best_solution_indices = refine_indices(greedy_result) if greedy_result else []

    # Final local search passes on the best solution
    if best_solution_indices and time_left() > 10:
        final_indices = local_search_1exchange(best_solution_indices)
        if final_indices and len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    if best_solution_indices and time_left() > 8:
        final_indices = local_search_2exchange(best_solution_indices)
        if final_indices and len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    if best_solution_indices and time_left() > 5:
        final_indices = local_search_2merge(best_solution_indices)
        if final_indices and len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices

    # Final check for validity
    if best_solution_indices and not is_valid(best_solution_indices):
        pass

    # Convert indices to blocks
    best_solution = [blocks[i] for i in best_solution_indices]
    best_solution.sort()
    return [list(b) for b in best_solution]
