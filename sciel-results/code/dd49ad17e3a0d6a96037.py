import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            tset_to_blocks[idx].append(i)
        block_masks[i] = mask

    full_mask = (1 << num_tsets) - 1
    
    # Precompute which tsets each block covers (as list of indices)
    block_tsets = [[] for _ in range(num_blocks)]
    for i, block in enumerate(all_blocks):
        block_tsets[i] = [tset_index[ts] for ts in itertools.combinations(block, t)]

    # Helper: Check if a set of blocks covers everything
    def is_valid_solution(blocks):
        covered = 0
        for b in blocks:
            covered |= block_masks[b]
        return covered == full_mask

    # Helper: Get cover counts
    def get_cover_counts(blocks):
        counts = [0] * num_tsets
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # 2. Greedy with priority queue for speed
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        # Build initial priority queue
        # Use negative count for min-heap
        heap = []
        for i in range(num_blocks):
            count = (block_masks[i] & uncovered).bit_count()
            if count > 0:
                # Add random tie-breaker
                heap.append((-count, rng.random(), i))
        heapq.heapify(heap)
        
        in_selected = [False] * num_blocks
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            
            # Pop until we find a block that still covers something uncovered
            best_idx = -1
            best_count = 0
            while heap:
                neg_count, _, i = heapq.heappop(heap)
                if in_selected[i]:
                    continue
                # Recompute actual coverage of uncovered
                actual_count = (block_masks[i] & uncovered).bit_count()
                if actual_count > 0:
                    best_idx = i
                    best_count = actual_count
                    break
            
            if best_idx == -1 or best_count == 0:
                break
                
            selected.append(best_idx)
            in_selected[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
            # Update heap: add blocks that may now be better
            # Only add blocks that cover at least one newly-relevant tset
            # For efficiency, just periodically rebuild or add candidates
            # Simple approach: add all blocks that cover any uncovered tset
            # But that's expensive. Instead, just continue with existing heap entries.
            # The heap may have stale entries but we recompute on pop.
            
        return selected

    # 3. Local Search with 1-block and 2-block swaps
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = get_cover_counts(sol)
        
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 80
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol): continue
                b = sol[pos]
                m = block_masks[b]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Update counts
                    m = block_masks[b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
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
                
            # Phase 2: 1-block swap
            if sol:
                found_swap = False
                attempts = 30
                if elapsed() < deadline - 30: attempts = 60
                
                for _ in range(attempts):
                    if not sol: break
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
                        # Redundant
                        m = block_masks[b_out]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list: continue
                    
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                            
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]: continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)
                            
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Update counts
                        m = block_masks[b_out]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        m = block_masks[b_in]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] += 1
                            m ^= lsb
                        in_sol[b_out] = False
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
            
            # Phase 3: 2-block swap (remove 2, add 2)
            if len(sol) >= 2:
                found_2swap = False
                attempts_2 = 10
                if elapsed() < deadline - 30: attempts_2 = 20
                
                for _ in range(attempts_2):
                    if len(sol) < 2: break
                    pos1 = rng.randint(0, len(sol) - 1)
                    pos2 = rng.randint(0, len(sol) - 1)
                    if pos1 == pos2: continue
                    
                    b_out1 = sol[pos1]
                    b_out2 = sol[pos2]
                    
                    # Combined critical mask: tsets covered only by b_out1 or b_out2
                    # A tset is critical if it's covered by exactly one of them (or both, but removing both would uncover it)
                    # Actually: tset is critical if cover_count == 1 and it's covered by b_out1 XOR b_out2, 
                    # or cover_count == 2 and covered by both b_out1 and b_out2
                    critical_mask = 0
                    m1 = block_masks[b_out1]
                    m2 = block_masks[b_out2]
                    # Tsets covered by exactly one of them with count 1
                    only1 = m1 & ~m2
                    only2 = m2 & ~m1
                    both = m1 & m2
                    m = only1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                    m = only2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                    m = both
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] <= 2:
                            critical_mask |= lsb
                        m ^= lsb
                    
                    if critical_mask == 0:
                        # Both redundant
                        for b in [b_out1, b_out2]:
                            m = block_masks[b]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] -= 1
                                m ^= lsb
                            in_sol[b] = False
                        # Remove from sol (be careful with indices)
                        if pos1 < pos2:
                            sol.pop(pos1)
                            sol.pop(pos2 - 1)
                        else:
                            sol.pop(pos2)
                            sol.pop(pos1 - 1)
                        improved = True
                        found_2swap = True
                        break
                    
                    # Find 2 blocks that cover all critical tsets
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                    
                    if len(crit_ts_list) > 4:
                        continue  # Too many critical tsets, unlikely to find good replacement
                    
                    # Try pairs of blocks from the smallest tset's candidates
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                    
                    candidates = [b for b in tset_to_blocks[min_ts_idx] if not in_sol[b]]
                    if len(candidates) > 50:
                        candidates = rng.sample(candidates, 50)
                    
                    found_pair = False
                    for i1 in range(len(candidates)):
                        b_in1 = candidates[i1]
                        m1 = block_masks[b_in1]
                        if (m1 & critical_mask) != critical_mask:
                            # Check if b_in1 covers all critical tsets alone
                            # If not, we need b_in2 to cover the rest
                            remaining = critical_mask & ~m1
                            if remaining == 0:
                                # b_in1 alone covers all critical
                                b_in2 = -1
                            else:
                                # Find b_in2 that covers remaining
                                # Use the smallest tset in remaining
                                rem_ts_list = []
                                m = remaining
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    rem_ts_list.append(idx)
                                    m ^= lsb
                                min_rem_idx = rem_ts_list[0]
                                min_rem_len = len(tset_to_blocks[min_rem_idx])
                                for ts_idx in rem_ts_list[1:]:
                                    curr_len = len(tset_to_blocks[ts_idx])
                                    if curr_len < min_rem_len:
                                        min_rem_idx = ts_idx
                                        min_rem_len = curr_len
                                found_b2 = False
                                for b2 in tset_to_blocks[min_rem_idx]:
                                    if in_sol[b2] or b2 == b_in1:
                                        continue
                                    if (block_masks[b2] & remaining) == remaining:
                                        b_in2 = b2
                                        found_b2 = True
                                        break
                                if not found_b2:
                                    continue
                        else:
                            b_in2 = -1
                        
                        # Apply the 2-block swap
                        # Remove b_out1, b_out2
                        for b in [b_out1, b_out2]:
                            m = block_masks[b]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] -= 1
                                m ^= lsb
                            in_sol[b] = False
                        
                        # Add b_in1, b_in2
                        for b in ([b_in1] if b_in2 == -1 else [b_in1, b_in2]):
                            m = block_masks[b]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] += 1
                                m ^= lsb
                            in_sol[b] = True
                        
                        # Update sol
                        # Remove b_out1 and b_out2 from sol
                        if pos1 < pos2:
                            sol.pop(pos1)
                            sol.pop(pos2 - 1)
                        else:
                            sol.pop(pos2)
                            sol.pop(pos1 - 1)
                        
                        # Add new blocks
                        if b_in2 == -1:
                            sol.append(b_in1)
                        else:
                            sol.append(b_in1)
                            sol.append(b_in2)
                        
                        found_2swap = True
                        found_pair = True
                        improved = True
                        break
                    
                    if found_2swap:
                        break
                
                if found_2swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            
            no_improve += 1
            
        return best_sol

    # 4. Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # Initial run
    sol = greedy_cover(0)
    ls_sol = local_search(sol, 0)
    best_solution = ls_sol
    best_size = len(ls_sol)
    
    # ILS Loop
    current_sol = best_solution
    rng_ils = random.Random(42)
    while elapsed() < deadline - 20:
        # Perturb: remove 2-4 random blocks and replace with random blocks
        perturbed = list(current_sol)
        
        # Remove random blocks
        num_remove = rng_ils.randint(2, min(4, len(perturbed)))
        for _ in range(num_remove):
            if not perturbed: break
            idx = rng_ils.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        
        # Fix coverage by adding blocks greedily
        covered = 0
        for b in perturbed:
            covered |= block_masks[b]
        uncovered = full_mask & ~covered
        
        # Add blocks greedily to fix
        while uncovered:
            if elapsed() > deadline - 10:
                break
            # Find block covering most uncovered
            best_idx = -1
            best_count = -1
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    
            if best_idx == -1 or best_count == 0:
                break
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        # Local Search on Perturbed
        ls_sol = local_search(perturbed, rng_ils.randint(0, 10**9))
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            current_sol = best_solution

    return [list(all_blocks[i]) for i in best_solution]
