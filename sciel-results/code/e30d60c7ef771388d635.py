import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build block_masks using integers (bitmasks) for fast set operations
    # Note: num_tsets for C(19,5) is 11628, which fits in a Python int
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        block_masks.append(mask)
        
    full_mask = (1 << num_tsets) - 1
    
    # Precompute for each tset, the list of blocks that cover it
    # This is crucial for fast candidate generation in swaps
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    # 2. Greedy Set Cover (Standard)
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count == num_tsets:
                        break
            
            if best_idx == -1 or best_count == 0:
                break
                
            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search (Deletion and 1-for-1 Swap)
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        
        # Precompute cover counts
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 200 # Increased patience
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Try to remove redundant blocks
            order = list(range(len(sol)))
            rng.shuffle(order)
            
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                mask = block_masks[b]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    update_counts(b, -1)
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
                # Try multiple random swaps per iteration to escape local minima
                attempts = 0
                max_attempts = 20
                while attempts < max_attempts:
                    attempts += 1
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                        
                    if critical_mask == 0:
                        update_counts(b_out, -1)
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        break
                        
                    # Find a block covering critical_mask
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list:
                        continue
                        
                    # Intersect lists of blocks covering each critical tset
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])
                        
                    sol_set = set(sol)
                    candidates -= sol_set
                    
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        update_counts(b_out, -1)
                        update_counts(b_in, 1)
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # 4. Simulated Annealing with 2-for-1 and 1-for-2 moves
    def simulated_annealing(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        
        # Cooling schedule
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1
        max_iterations = 5000
        
        for it in range(max_iterations):
            if elapsed() > deadline - 15:
                break
            if T < min_T:
                break
                
            move_type = rng.random()
            
            if move_type < 0.4:
                # Swap: 1-for-1
                if not sol:
                    continue
                pos = rng.randint(0, len(sol) - 1)
                b_out = sol[pos]
                
                critical_mask = 0
                m = block_masks[b_out]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                    
                if critical_mask == 0:
                    # Just remove
                    update_counts(b_out, -1)
                    in_sol[b_out] = False
                    sol.pop(pos)
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue

                crit_ts_list = []
                m = critical_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    crit_ts_list.append(idx)
                    m ^= lsb
                    
                candidates = set(tset_to_blocks[crit_ts_list[0]])
                for ts_idx in crit_ts_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(tset_to_blocks[ts_idx])
                    
                sol_set = set(sol)
                candidates -= sol_set
                
                if not candidates:
                    continue
                    
                b_in = rng.choice(list(candidates))
                
                # Apply swap
                update_counts(b_out, -1)
                in_sol[b_out] = False
                update_counts(b_in, 1)
                in_sol[b_in] = True
                sol[pos] = b_in
                
                # Size unchanged, always accept to explore
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_sol = list(sol)
                    
            elif move_type < 0.7:
                # 2-for-1: Remove 2 blocks, add 1
                if len(sol) < 2:
                    continue
                idx1, idx2 = rng.sample(range(len(sol)), 2)
                b1 = sol[idx1]
                b2 = sol[idx2]
                
                # Check if removing both is "almost" valid or if we can find a replacement
                # We need a block B that covers all tsets that become uncovered
                # Tsets covered only by b1 or b2 (or both)
                mask1 = block_masks[b1]
                mask2 = block_masks[b2]
                
                # Tsets that become uncovered if we remove b1 and b2
                # A tset is uncovered if it was covered ONLY by b1 or ONLY by b2 (and not by others)
                # Actually, if a tset is covered by b1 and b2, removing both uncovers it.
                # If covered by b1 and b3, removing b1 leaves it covered by b3.
                
                # Critical tsets for the pair {b1, b2}:
                # Tsets where cover_count == 1 and covered by b1 OR cover_count == 1 and covered by b2
                # Wait, if a tset is covered by b1 and b2, cover_count is at least 2.
                # If we remove both, cover_count drops by 2.
                # So a tset becomes uncovered if:
                # 1. It is covered by b1, not by b2, and cover_count == 1
                # 2. It is covered by b2, not by b1, and cover_count == 1
                # 3. It is covered by both b1 and b2, and cover_count == 2
                
                critical_mask = 0
                m1 = mask1
                m2 = mask2
                m_both = m1 & m2
                m1_only = m1 & ~m2
                m2_only = m2 & ~m1
                
                # Check m1_only
                m = m1_only
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                    
                # Check m2_only
                m = m2_only
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                        
                # Check m_both
                m = m_both
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 2:
                        critical_mask |= lsb
                    m ^= lsb
                
                if critical_mask == 0:
                    # Both are redundant? Just remove one or both?
                    # If critical_mask is 0, it means no tset becomes uncovered.
                    # This implies both blocks are redundant in the context of the other?
                    # Actually, if critical_mask is 0, we can remove both.
                    update_counts(b1, -1)
                    update_counts(b2, -1)
                    in_sol[b1] = False
                    in_sol[b2] = False
                    # Remove from sol (handle indices carefully)
                    sol.pop(max(idx1, idx2))
                    sol.pop(min(idx1, idx2))
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue

                crit_ts_list = []
                m = critical_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    crit_ts_list.append(idx)
                    m ^= lsb
                    
                candidates = set(tset_to_blocks[crit_ts_list[0]])
                for ts_idx in crit_ts_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(tset_to_blocks[ts_idx])
                    
                sol_set = set(sol)
                candidates -= sol_set
                candidates -= {b1, b2} # Don't swap with the blocks we are removing
                
                if not candidates:
                    continue
                    
                b_in = rng.choice(list(candidates))
                
                # Apply 2-for-1
                update_counts(b1, -1)
                update_counts(b2, -1)
                in_sol[b1] = False
                in_sol[b2] = False
                update_counts(b_in, 1)
                in_sol[b_in] = True
                
                # Remove b1, b2 from sol, add b_in
                sol.pop(max(idx1, idx2))
                sol.pop(min(idx1, idx2))
                sol.append(b_in)
                
                delta_e = 1 # Size increased by 1 (removed 2, added 1 -> net -1? No, removed 2 added 1 is -1)
                # Wait. Size was N. Removed 2, added 1. New size N-1.
                # Delta = -1. Improvement.
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_sol = list(sol)
                    
            else:
                # 1-for-2: Remove 1 block, add 2
                if not sol:
                    continue
                pos = rng.randint(0, len(sol) - 1)
                b_out = sol[pos]
                
                critical_mask = 0
                m = block_masks[b_out]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                    
                if critical_mask == 0:
                    # Just remove
                    update_counts(b_out, -1)
                    in_sol[b_out] = False
                    sol.pop(pos)
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue

                crit_ts_list = []
                m = critical_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    crit_ts_list.append(idx)
                    m ^= lsb
                    
                # We need 2 blocks that together cover critical_mask
                # This is harder to sample directly.
                # Strategy: Pick one block covering a random critical tset, then pick another covering the rest?
                # Or just try to find a pair.
                # For simplicity, let's skip complex 1-for-2 sampling and rely on 2-for-1 and swaps.
                # Or, pick b1 from candidates of first crit tset, then b2 from candidates of remaining crit tsets.
                
                if len(crit_ts_list) == 0:
                    continue
                    
                b1_candidates = set(tset_to_blocks[crit_ts_list[0]])
                sol_set = set(sol)
                b1_candidates -= sol_set
                b1_candidates -= {b_out}
                
                if not b1_candidates:
                    continue
                    
                b1 = rng.choice(list(b1_candidates))
                
                # Now find b2 that covers (critical_mask & ~block_masks[b1])
                remaining_mask = critical_mask & ~block_masks[b1]
                
                if remaining_mask == 0:
                    # b1 covers all critical tsets. We can just swap 1-for-1.
                    # But we are in 1-for-2 mode. Let's just do a swap.
                    update_counts(b_out, -1)
                    in_sol[b_out] = False
                    update_counts(b1, 1)
                    in_sol[b1] = True
                    sol[pos] = b1
                    continue
                    
                rem_ts_list = []
                m = remaining_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    rem_ts_list.append(idx)
                    m ^= lsb
                    
                if not rem_ts_list:
                    continue
                    
                b2_candidates = set(tset_to_blocks[rem_ts_list[0]])
                b2_candidates -= sol_set
                b2_candidates -= {b_out, b1}
                
                if not b2_candidates:
                    continue
                    
                b2 = rng.choice(list(b2_candidates))
                
                # Verify b1 and b2 together cover critical_mask
                if (block_masks[b1] | block_masks[b2]) & critical_mask != critical_mask:
                    continue
                    
                # Apply 1-for-2
                update_counts(b_out, -1)
                in_sol[b_out] = False
                update_counts(b1, 1)
                in_sol[b1] = True
                update_counts(b2, 1)
                in_sol[b2] = True
                
                sol.pop(pos)
                sol.append(b1)
                sol.append(b2)
                
                # Size increased by 1. Accept with probability exp(-1/T)
                if rng.random() < math.exp(-1.0 / T):
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                else:
                    # Revert
                    update_counts(b1, -1)
                    update_counts(b2, -1)
                    in_sol[b1] = False
                    in_sol[b2] = False
                    update_counts(b_out, 1)
                    in_sol[b_out] = True
                    sol.pop()
                    sol.pop()
                    sol.append(b_out)
                    
            T *= cooling_rate

        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    for seed in range(num_restarts):
        if elapsed() > deadline - 15:
            break
        sol = greedy_cover(seed)
        if not sol:
            continue
            
        # Local search
        ls_sol = local_search(sol, seed)
        
        # Simulated Annealing
        sa_sol = simulated_annealing(ls_sol, seed)
        
        if len(sa_sol) < best_size:
            best_size = len(sa_sol)
            best_solution = sa_sol
            
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
