import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0 # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks and their bitmasks
    # Optimization: Pre-compute list of tset indices for each block to speed up local search
    all_blocks = []
    block_masks = []
    block_tsets = [] # List of lists of tset indices covered by each block
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        block_ts_list = []
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            block_ts_list.append(idx)
        all_blocks.append(block)
        block_masks.append(mask)
        block_tsets.append(block_ts_list)
        
    num_blocks = len(all_blocks)
    
    # Initial full mask
    full_mask = (1 << num_tsets) - 1

    # 2. Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            # To speed up, we can sort blocks initially by size (all same) or random shuffle
            # But we need to pick the best one each time.
            
            # Optimization: Keep a list of candidate indices.
            # If a block is selected, remove it.
            # Scanning all blocks each time is O(N_blocks * N_iterations).
            # 75k * 300 = 22.5M. Acceptable.
            
            # We will iterate through all blocks and pick the one with max new coverage.
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count == num_tsets: # Perfect cover (unlikely)
                        break
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search (1-for-1 swap and deletion)
    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)
            
    def local_search_v2(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                    
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                # Increase swap attempts if we have time
                max_swap_attempts = 20
                if elapsed() < deadline - 30:
                    max_swap_attempts = 50
                    
                for _ in range(max_swap_attempts):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    critical_mask = 0
                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            crit_ts_list.append(ts_idx)
                        
                    if critical_mask == 0:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    if not crit_ts_list:
                        continue
                        
                    # Optimization: Find the critical tset with the fewest covering blocks
                    # This minimizes the candidate pool we need to filter
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                            
                    # Filter candidates from the shortest list
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        # Check if this block covers ALL critical tsets
                        # (block_masks[b_idx] & critical_mask) == critical_mask
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)
                            
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Perform swap
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run greedy restarts
    # Adjust number of restarts based on time taken for first one
    # Increased restarts to take advantage of faster local search
    num_restarts = 8
    for seed in range(num_restarts):
        if elapsed() > deadline - 20:
            break
            
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    if best_solution is None:
        # Fallback: return empty or raise
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
