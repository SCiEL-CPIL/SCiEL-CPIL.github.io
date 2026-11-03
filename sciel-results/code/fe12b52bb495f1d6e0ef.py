import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices (masks)
    # For v=24, k=7, this is C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    
    block_list = []
    block_masks = []
    block_triple_indices = [] # Store list of triple indices for each block
    
    # Optimization: Precompute bit for each triple
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i
        
    # Generate all blocks and their masks
    for block in itertools.combinations(range(v), k):
        mask = 0
        triple_idxs = []
        for sub in itertools.combinations(block, t):
            idx = subset_index[sub]
            mask |= triple_bit[sub]
            triple_idxs.append(idx)
        block_list.append(block)
        block_masks.append(mask)
        block_triple_indices.append(triple_idxs)
        
    n_blocks = len(block_list)
    
    # --- Search Strategy: Weighted Greedy + Criticality-Based Local Search ---
    
    def get_greedy_solution(rng_seed):
        rng = random.Random(rng_seed)
        uncovered_mask = (1 << n_subsets) - 1
        selected_indices = []
        used_indices = set()
        
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break
                
            chosen_idx = rng.choice(best_candidates)
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    def local_search(initial_blocks, rng, time_limit):
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Compute coverage counts
        coverage_count = [0] * n_subsets
        for b_idx in current_blocks:
            for t_idx in block_triple_indices[b_idx]:
                coverage_count[t_idx] += 1
        
        best_count = len(current_blocks)
        best_blocks = list(current_blocks)
        
        # Criticality-Based Local Search
        # 1. Try to delete redundant blocks (where all covered triples have count > 1)
        # 2. If stuck, try swaps to escape local minima
        # 3. Use criticality analysis: prioritize blocks that cover many "critical" triples (count=1)
        
        no_improvement_count = 0
        max_no_improvement = 200
        
        while time.time() < time_limit and no_improvement_count < max_no_improvement:
            improved = False
            
            # Phase 1: Try to delete redundant blocks
            # Find all redundant blocks
            redundant_blocks = []
            for b_idx in current_blocks:
                is_redundant = True
                for t_idx in block_triple_indices[b_idx]:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(b_idx)
            
            if redundant_blocks:
                # Remove one redundant block (random choice for diversity)
                b_idx = rng.choice(redundant_blocks)
                current_blocks.remove(b_idx)
                current_set.remove(b_idx)
                for t_idx in block_triple_indices[b_idx]:
                    coverage_count[t_idx] -= 1
                
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                improved = True
                no_improvement_count = 0
                continue
            
            # Phase 2: No redundant blocks found. Try swaps.
            # Criticality analysis: For each block, count how many critical triples (count=1) it covers.
            # A block covering many critical triples is "essential" - removing it would break coverage.
            # We want to swap out blocks that cover few critical triples with blocks that cover more.
            
            # Compute criticality for each current block
            block_criticality = {}
            for b_idx in current_blocks:
                crit_count = 0
                for t_idx in block_triple_indices[b_idx]:
                    if coverage_count[t_idx] == 1:
                        crit_count += 1
                block_criticality[b_idx] = crit_count
            
            # Try to find a swap: remove block A (low criticality), add block B (high gain)
            # Sort current blocks by criticality (ascending) - try to remove less critical ones first
            sorted_current = sorted(current_blocks, key=lambda x: block_criticality[x])
            
            swap_found = False
            for b_remove in sorted_current[:10]:  # Try top 10 least critical blocks
                if time.time() >= time_limit:
                    break
                    
                # For this b_remove, find a good b_add
                # b_add should cover triples that b_remove covers, plus maybe more
                # And b_add should not make any triple uncovered
                
                # Triples in b_remove that have count > 1 can be "freed"
                # Triples in b_remove that have count == 1 must be covered by b_add
                
                required_triples = []
                for t_idx in block_triple_indices[b_remove]:
                    if coverage_count[t_idx] == 1:
                        required_triples.append(t_idx)
                
                # Find b_add that covers all required_triples
                # And has high "gain" (covers many triples not currently well-covered)
                
                best_b_add = -1
                best_score = -1
                
                # Sample some candidate blocks to avoid O(N) scan
                candidates = rng.sample(range(n_blocks), min(1000, n_blocks))
                
                for b_add in candidates:
                    if b_add in current_set:
                        continue
                    
                    # Check if b_add covers all required_triples
                    valid = True
                    for t_idx in required_triples:
                        if t_idx not in block_triple_indices[b_add]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: number of triples in b_add with count <= 2 (would benefit from extra coverage)
                    # Plus bonus for covering triples with count == 1 (if any remain)
                    score = 0
                    for t_idx in block_triple_indices[b_add]:
                        if coverage_count[t_idx] <= 2:
                            score += 1
                        if coverage_count[t_idx] == 1:
                            score += 5  # Bonus for covering critical triples
                    
                    if score > best_score:
                        best_score = score
                        best_b_add = b_add
                
                if best_b_add != -1:
                    # Perform swap
                    current_blocks.remove(b_remove)
                    current_set.remove(b_remove)
                    for t_idx in block_triple_indices[b_remove]:
                        coverage_count[t_idx] -= 1
                    
                    current_blocks.append(best_b_add)
                    current_set.add(best_b_add)
                    for t_idx in block_triple_indices[best_b_add]:
                        coverage_count[t_idx] += 1
                    
                    swap_found = True
                    improved = True
                    no_improvement_count = 0
                    break
            
            if not swap_found:
                no_improvement_count += 1
                # If stuck for a while, try a random swap to escape
                if no_improvement_count > 50 and rng.random() < 0.3:
                    if current_blocks:
                        b_remove = rng.choice(current_blocks)
                        # Find any valid b_add
                        required_triples = []
                        for t_idx in block_triple_indices[b_remove]:
                            if coverage_count[t_idx] == 1:
                                required_triples.append(t_idx)
                        
                        for _ in range(50):
                            b_add = rng.randrange(n_blocks)
                            if b_add in current_set:
                                continue
                            valid = True
                            for t_idx in required_triples:
                                if t_idx not in block_triple_indices[b_add]:
                                    valid = False
                                    break
                            if valid:
                                # Perform swap
                                current_blocks.remove(b_remove)
                                current_set.remove(b_remove)
                                for t_idx in block_triple_indices[b_remove]:
                                    coverage_count[t_idx] -= 1
                                
                                current_blocks.append(b_add)
                                current_set.add(b_add)
                                for t_idx in block_triple_indices[b_add]:
                                    coverage_count[t_idx] += 1
                                improved = True
                                break
            
            if not improved:
                no_improvement_count += 1
        
        return best_blocks

    # Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts of Greedy + Local Search
    restart_id = 0
    while time.time() < deadline - 10.0:
        # 1. Generate greedy solution
        init_blocks = get_greedy_solution(restart_id)
        
        # 2. Local search
        ls_start = time.time()
        ls_limit = min(ls_start + 5.0, deadline - 2.0) # Give 5s to local search per restart
        refined_blocks = local_search(init_blocks, random.Random(restart_id), ls_limit)
        
        count = len(refined_blocks)
        if count < best_count:
            best_count = count
            best_blocks = [block_list[i] for i in refined_blocks]
            
        restart_id += 1
        
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return []
