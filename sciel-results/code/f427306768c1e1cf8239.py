<<<<<<< SEARCH
    # Run multiple greedy restarts
    best_selected = None
    best_len = float('inf')

    configs = [
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('random', 2024),
        ('exp', 314),
        ('random', 4242),
        ('exp', 2718),
    ]

    initial_solutions = []
    for i, (mode, seed) in enumerate(configs):
        if time_left() < 25:
            break
        sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]
        initial_solutions.append(sel)
    
    # Additional restarts with best config if time permits
    if time_left() > 40 and initial_solutions:
        # Find the best initial config
        best_init = min(initial_solutions, key=len)
        best_init_mode = 'random'
        for extra_seed in [9999, 12345, 67890]:
            if time_left() < 20:
                break
            sel = run_greedy(best_init_mode, extra_seed)
            sel = local_search(sel, 0.10)
            if len(sel) < best_len:
                best_len = len(sel)
                best_selected = sel[:]
            initial_solutions.append(sel)

    # Run genetic algorithm with the initial solutions
    if time_left() > 35 and initial_solutions:
        ga_result = genetic_algorithm(initial_solutions, 0.3)
        if ga_result and len(ga_result) < best_len:
            best_len = len(ga_result)
            best_selected = ga_result

    # Final polish: multiple local search passes
    if best_selected and time_left() > 15:
        for polish_pass in range(3):
            if time_left() < 10:
                break
            best_selected = local_search(best_selected, 0.20)
            if len(best_selected) < best_len:
                best_len = len(best_selected)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
    
    # Deduplicate blocks
    seen = set()
    unique_result = []
    for block in result:
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            unique_result.append(block)
    
    return unique_result
=======
    # Run multiple greedy restarts
    best_selected = None
    best_len = float('inf')

    configs = [
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('random', 2024),
        ('exp', 314),
        ('random', 4242),
        ('exp', 2718),
        ('random', 31415),
        ('inverse', 27182),
        ('exp', 16180),
        ('random', 82718),
    ]

    initial_solutions = []
    for i, (mode, seed) in enumerate(configs):
        if time_left() < 20:
            break
        sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]
        initial_solutions.append(sel)
    
    # Simulated Annealing phase to escape local minima
    if best_selected and time_left() > 30:
        rng_sa = np.random.default_rng(42)
        current = list(best_selected)
        current_len = len(current)
        best_sa = list(current)
        best_sa_len = current_len
        
        # Compute cover counts for current
        cover_count_sa = np.zeros(num_t, dtype=np.int32)
        for i in current:
            for idx in block_t_subsets[i]:
                cover_count_sa[idx] += 1
        current_set = set(current)
        
        T = 5.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        max_iter = 5000
        
        for it in range(max_iter):
            if time_left() < 15:
                break
            if T < T_min:
                break
            
            # Random move: swap one block or drop one block
            move_type = rng_sa.random()
            
            if move_type < 0.7 and current:
                # Swap a block
                si = int(rng_sa.integers(0, len(current)))
                drop_block = current[si]
                drop_set = block_t_subsets[drop_block]
                
                # Find a random candidate to replace
                # Prefer blocks that cover critical t-subsets
                critical = [idx for idx in drop_set if cover_count_sa[idx] == 1]
                if critical:
                    # Pick from blocks covering all critical
                    candidate_sets = [set(t_to_blocks[idx]) for idx in critical[:3]]  # Limit to 3 critical for speed
                    candidates = candidate_sets[0]
                    for cs in candidate_sets[1:]:
                        candidates.intersection_update(cs)
                    candidates -= current_set
                    if not candidates:
                        continue
                    new_block = int(rng_sa.choice(list(candidates)))
                else:
                    # Random block not in current
                    all_blocks = list(range(num_k))
                    candidates = [b for b in all_blocks if b not in current_set]
                    if not candidates:
                        continue
                    new_block = int(rng_sa.choice(candidates))
                
                # Apply swap
                for idx in drop_set:
                    cover_count_sa[idx] -= 1
                for idx in block_t_subsets[new_block]:
                    cover_count_sa[idx] += 1
                current[si] = new_block
                current_set.discard(drop_block)
                current_set.add(new_block)
                
                # Compute delta (change in number of uncovered t-subsets)
                delta = 0
                for idx in drop_set:
                    if cover_count_sa[idx] == 0:
                        delta += 1
                for idx in block_t_subsets[new_block]:
                    if cover_count_sa[idx] == 1:
                        delta -= 1
                
                # Accept with probability
                if delta <= 0 or rng_sa.random() < math.exp(-delta / T):
                    if len(current) < best_sa_len:
                        best_sa_len = len(current)
                        best_sa = list(current)
                else:
                    # Reject: revert
                    for idx in block_t_subsets[new_block]:
                        cover_count_sa[idx] -= 1
                    for idx in drop_set:
                        cover_count_sa[idx] += 1
                    current[si] = drop_block
                    current_set.discard(new_block)
                    current_set.add(drop_block)
                    
            elif move_type < 0.9 and current and len(current) > 100:
                # Drop a block (if it doesn't cause uncovered)
                si = int(rng_sa.integers(0, len(current)))
                drop_block = current[si]
                drop_set = block_t_subsets[drop_block]
                
                # Check if safe to drop
                can_drop = True
                for idx in drop_set:
                    if cover_count_sa[idx] == 1:
                        can_drop = False
                        break
                
                if can_drop:
                    for idx in drop_set:
                        cover_count_sa[idx] -= 1
                    current.pop(si)
                    current_set.discard(drop_block)
                    if len(current) < best_sa_len:
                        best_sa_len = len(current)
                        best_sa = list(current)
                else:
                    # Try to drop anyway and accept with probability
                    for idx in drop_set:
                        cover_count_sa[idx] -= 1
                    current.pop(si)
                    current_set.discard(drop_block)
                    
                    delta = 0
                    for idx in drop_set:
                        if cover_count_sa[idx] == 0:
                            delta += 1
                    
                    if delta <= 0 or rng_sa.random() < math.exp(-delta / T):
                        if len(current) < best_sa_len:
                            best_sa_len = len(current)
                            best_sa = list(current)
                    else:
                        # Revert
                        for idx in drop_set:
                            cover_count_sa[idx] += 1
                        current.append(drop_block)
                        current_set.add(drop_block)
            
            T *= cooling
        
        # Final local search on best SA result
        if best_sa and len(best_sa) <= best_len:
            best_sa = local_search(best_sa, 0.15)
            if len(best_sa) < best_len:
                best_len = len(best_sa)
                best_selected = best_sa[:]
    
    # Additional restarts if time permits
    if time_left() > 25 and initial_solutions:
        for extra_seed in [9999, 12345, 67890, 54321]:
            if time_left() < 15:
                break
            sel = run_greedy('random', extra_seed)
            sel = local_search(sel, 0.10)
            if len(sel) < best_len:
                best_len = len(sel)
                best_selected = sel[:]
            initial_solutions.append(sel)

    # Final polish: multiple local search passes
    if best_selected and time_left() > 12:
        for polish_pass in range(4):
            if time_left() < 8:
                break
            best_selected = local_search(best_selected, 0.18)
            if len(best_selected) < best_len:
                best_len = len(best_selected)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
    
    # Deduplicate blocks
    seen = set()
    unique_result = []
    for block in result:
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            unique_result.append(block)
    
    return unique_result
>>>>>>> REPLACE
