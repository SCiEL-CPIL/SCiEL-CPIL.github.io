import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Generate all candidate blocks and their masks
    # For v=24, k=7, num_blocks = 346104. This is manageable.
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
    
    # Precompute block lists for each triple for fast intersection lookups
    # triple_to_blocks: list of indices of blocks containing this triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        # Iterate set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb
            
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                return None # Timeout
            
            best_gain = -1
            candidates = []
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                return selected_indices # Should not happen if valid input
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    def refine(selected_indices):
        if selected_indices is None:
            return []
            
        solution_set = set(selected_indices)
        
        # Calculate coverage counts
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
            if time.time() > deadline:
                break
            changed = False
            # Iterate over a copy of the list to allow modification
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
                    
        return [blocks[i] for i in solution_set]

    def local_search_2exchange(solution_indices):
        """
        Attempts to reduce the number of blocks by replacing pairs of blocks
        with a single block that covers all 'unique' triples of the pair.
        """
        if solution_indices is None:
            return []
            
        sol_set = set(solution_indices)
        if not sol_set:
            return []
            
        # Precompute masks for current solution blocks for fast union/diff
        # We will maintain coverage counts and a "unique" check logic
        
        # To speed up "find intersection of block lists for a set of triples",
        # we need the triples that are uniquely covered by the pair.
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Recalculate coverage counts and unique status
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            # List of blocks in solution to iterate
            sol_list = list(sol_set)
            
            # Check all pairs
            for i in range(len(sol_list)):
                for j in range(i + 1, len(sol_list)):
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    # Find triples covered by b1 or b2 but NOT by any other block in sol_set
                    # This is (Mask[b1] | Mask[b2]) & ~(Union of others)
                    # Efficiently: 
                    # A triple is "unique to pair" if coverage_count[t] == 2 and t is in b1 or b2.
                    # Wait, if coverage_count[t] == 2, it is covered by exactly 2 blocks.
                    # If those 2 blocks are b1 and b2, then it is unique to the pair.
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    # Identify triples in mask_pair with coverage_count == 2
                    # We need to ensure they are covered by b1 and b2.
                    # Since coverage_count is global, if count==2 and t in mask_pair, 
                    # it could be covered by b1 and b3, not b2.
                    # So we must check if t is in mask1 XOR mask2? No.
                    # t is unique to pair {b1, b2} iff t is in (mask1 | mask2) 
                    # AND t is NOT in any other block in sol_set.
                    # This is equivalent to: coverage_count[t] == 2 AND t in mask1 AND t in mask2? 
                    # No, if t is in mask1 and mask3, count is 2, but it's not in mask2.
                    # So we need: t in mask1 OR t in mask2, AND coverage_count[t] == 2, 
                    # AND (t in mask1 XOR t in mask2 is not sufficient).
                    # Actually, if t is in mask1 and mask2, and count is 2, then it is ONLY in b1 and b2.
                    # If t is in mask1 but not mask2, and count is 2, it is in b1 and some b_k (k!=2). 
                    # In that case, removing b1 and b2 would leave t uncovered (unless b_k covers it, but b_k is not b2).
                    # So we only care about triples that are covered by BOTH b1 and b2?
                    # No. If t is covered by b1 and b3, and we remove b1 and b2, t is still covered by b3. 
                    # So t is NOT critical for the pair removal.
                    # The critical triples are those covered by b1 or b2, but NOT by any other block in sol_set.
                    # These are exactly the triples where coverage_count[t] == 1 (if only in one) 
                    # or coverage_count[t] == 2 (if in both b1 and b2).
                    # Wait, if t is in b1 only, count=1. Critical.
                    # If t is in b1 and b2, count=2. Critical.
                    # If t is in b1 and b3, count=2. Not critical for pair {b1,b2} because b3 keeps it.
                    
                    # So, Critical Triples = {t | t in (mask1 | mask2) AND (coverage_count[t] == 1 OR (coverage_count[t] == 2 AND t in mask1 AND t in mask2))}
                    
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        if coverage_count[idx_t] == 1:
                            critical_triples.append(idx_t)
                        elif coverage_count[idx_t] == 2:
                            # Check if in both
                            if (mask1 & lsb) and (mask2 & lsb):
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        # Both blocks are redundant relative to each other? 
                        # If no critical triples, we can remove one? 
                        # If mask1 subset of others and mask2 subset of others, we remove both.
                        # But usually redundancy removal handles singletons.
                        # If we remove both, we save 2 blocks.
                        # Check if we can remove both.
                        # If critical_triples is empty, it means every triple in b1 is covered by >=1 other block,
                        # and every triple in b2 is covered by >=1 other block.
                        # So we can remove both.
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break # Restart loop with new state
                    else:
                        # Find a block b_k that covers all critical_triples
                        # Intersection of triple_to_blocks[t] for t in critical_triples
                        # Use the smallest list to start
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        # Get candidates from the first triple
                        candidates = triple_to_blocks[critical_triples[0]]
                        
                        # Intersect with others
                        # Optimization: if len(critical_triples) > 3, intersection is likely empty.
                        # But let's try.
                        for t_idx in critical_triples[1:]:
                            if not candidates:
                                break
                            # Intersect candidates with triple_to_blocks[t_idx]
                            # Convert to set for speed? Or just loop.
                            # Lists are sorted? Not necessarily.
                            # Use set intersection.
                            candidates = set(candidates).intersection(triple_to_blocks[t_idx])
                            candidates = list(candidates)
                            
                        if candidates:
                            # Check if any candidate is NOT in sol_set
                            for b_k in candidates:
                                if b_k not in sol_set:
                                    # Replace b1, b2 with b_k
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_k)
                                    improved = True
                                    break # Restart loop
                            
                            if improved:
                                break
                            
            # If we finished the double loop without 'improved', we break the while loop
            # Note: If we removed blocks, we break out of inner loops and restart while.
            
        return [blocks[i] for i in sol_set]

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 5 # Reduced to allow time for local search
    for i in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s buffer for local search
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        # Refine (remove redundant)
        refined = refine(greedy_result)
        
        # Local Search (2-exchange)
        # Convert refined list back to indices for local search
        # refine returns list of blocks (tuples). We need indices.
        # It's easier to keep indices in refine.
        
        if len(refined) < best_size:
            best_size = len(refined)
            # Store indices for local search
            # We need to map blocks back to indices.
            # Since blocks are unique tuples, we can use a dict.
            # But we already have `blocks` list.
            # Let's modify refine to return indices.
            
    # Redefine refine to return indices to facilitate local search
    def refine_indices(selected_indices):
        if selected_indices is None:
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
            if time.time() > deadline:
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

    best_solution_indices = None
    best_size = float('inf')
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        refined_indices = refine_indices(greedy_result)
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution_indices = refined_indices
            
        # Run local search on the best one found so far? 
        # Or on this one?
        # Let's run local search on the best one found at the end.
        
    if best_solution_indices is not None:
        # Run local search on the best greedy result
        # This might take time, so we do it once on the best candidate
        final_indices = local_search_2exchange(best_solution_indices)
        if len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    if best_solution_indices is None:
        # Fallback
        greedy_result = run_greedy(0)
        best_solution_indices = refine_indices(greedy_result) if greedy_result else []

    # Convert indices to blocks
    best_solution = [blocks[i] for i in best_solution_indices]
    best_solution.sort()
    return [list(b) for b in best_solution]
