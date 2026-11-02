<<<<<<< SEARCH
    # Phase 1: Standard greedy with multiple seeds
    restart_count = 0
    while time.time() - start_time < time_limit - 35.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=False)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    # Phase 2: Column generation with dual-guided swaps
    if time.time() - start_time < time_limit - 25.0:
        for seed in range(10):
            if time.time() - start_time > time_limit - 25.0:
                break
            blocks, score = column_generation(seed, max_cg_iters=30)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 3: Randomized weighted greedy
    if time.time() - start_time < time_limit - 15.0:
        for seed in range(5):
            if time.time() - start_time > time_limit - 15.0:
                break
            blocks, score = run_greedy(2000 + seed, use_random_weights=True)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks
=======
    def greedy_packing(seed, lambda_max=1, max_iters=500):
        """Greedy Packing with Conflict Resolution: select max 7-subsets with
        no 3-subset covered more than lambda_max times, then complete to a valid
        covering and minimize."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Track coverage count for each t-subset
        coverage = np.zeros(n_ts, dtype=np.int32)
        selected = []
        selected_set = set()
        
        # Precompute conflict structure: for each block, which other blocks share a t-subset
        # We'll track this dynamically
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 12.0:
                break
            
            # Find all blocks that don't violate lambda_max constraint
            # A block b is "feasible" if for all t-subs in b, coverage[sub] < lambda_max
            # Vectorized: for each block, check max coverage among its t-subs
            block_max_cov = coverage[block_t_indices].max(axis=1)
            feasible_mask = block_max_cov < lambda_max
            
            # Among feasible blocks, prefer those that add the most NEW coverage
            # (t-subs not yet covered at all)
            new_cov_count = (coverage[block_t_indices] == 0).sum(axis=1)
            
            # Also consider: blocks that cover t-subs with lowest coverage (spread)
            # This helps build a balanced packing
            avg_cov = coverage[block_t_indices].mean(axis=1)
            
            # Score: prioritize new coverage, then low average coverage
            feasible_indices = np.where(feasible_mask)[0]
            if len(feasible_indices) == 0:
                break
            
            # Among feasible, pick one with highest new coverage, tie-break by lowest avg cov
            scores = new_cov_count[feasible_indices] * 1000.0 - avg_cov[feasible_indices]
            best_local = np.argmax(scores)
            chosen = feasible_indices[best_local]
            
            # Additional randomization for diversity
            if len(feasible_indices) > 1 and random.random() < 0.3:
                # Pick randomly from top 20%
                top_frac = max(1, len(feasible_indices) // 5)
                top_indices = feasible_indices[np.argsort(scores)[-top_frac:]]
                chosen = top_indices[random.randint(0, len(top_indices) - 1)]
            
            if chosen in selected_set:
                # Try to find another feasible block
                feasible_indices2 = np.delete(feasible_indices, np.where(feasible_indices == chosen)[0])
                if len(feasible_indices2) == 0:
                    break
                chosen = feasible_indices2[random.randint(0, len(feasible_indices2) - 1)]
            
            selected.append(chosen)
            selected_set.add(chosen)
            coverage[block_t_indices[chosen]] += 1
        
        # Now we have a packing. Check if it's already a covering.
        if np.all(coverage >= 1):
            # It's a valid covering! Minimize it.
            current_blocks = list(selected)
            coverage_count = coverage.copy()
            current_blocks = aggressive_deletion(current_blocks, coverage_count)
            coverage_count = rebuild_coverage(current_blocks)
            
            if time.time() - start_time < time_limit - 10.0:
                current_blocks = local_search(current_blocks, coverage_count, max_iters=200)
                coverage_count = rebuild_coverage(current_blocks)
                current_blocks = aggressive_deletion(current_blocks, coverage_count)
            
            return current_blocks, len(current_blocks)
        else:
            # Not a covering yet. Complete it with greedy set cover on remaining uncovered.
            uncovered_mask = coverage < 1
            n_uncovered = np.sum(uncovered_mask)
            
            if n_uncovered == 0:
                current_blocks = list(selected)
                coverage_count = coverage.copy()
                current_blocks = aggressive_deletion(current_blocks, coverage_count)
                coverage_count = rebuild_coverage(current_blocks)
                return current_blocks, len(current_blocks)
            
            # Greedy complete: add blocks covering most uncovered triples
            blocks_indices = list(selected)
            while n_uncovered > 0:
                if time.time() - start_time > time_limit - 12.0:
                    break
                
                # For each block, count how many uncovered t-subs it covers
                gains = uncovered_mask[block_t_indices].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                
                max_gain_blocks = np.where(gains == max_gain)[0]
                if len(max_gain_blocks) > 30:
                    candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
                else:
                    candidates = list(max_gain_blocks)
                
                # Among candidates, prefer ones that add the most NEW coverage overall
                best_candidate = candidates[0]
                best_new = -1
                for cand in candidates:
                    newly = (~uncovered_mask[block_t_indices[cand]] & (coverage[block_t_indices[cand]] == 0)).sum()
                    if newly > best_new:
                        best_new = newly
                        best_candidate = cand
                
                blocks_indices.append(best_candidate)
                subs = block_t_indices[best_candidate]
                newly_uncovered = uncovered_mask[subs]
                n_uncovered -= np.sum(newly_uncovered)
                coverage[subs] += 1
                uncovered_mask[subs] = coverage[subs] < 1
            
            current_blocks = list(blocks_indices)
            coverage_count = rebuild_coverage(current_blocks)
            current_blocks = aggressive_deletion(current_blocks, coverage_count)
            coverage_count = rebuild_coverage(current_blocks)
            
            if time.time() - start_time < time_limit - 10.0:
                current_blocks = local_search(current_blocks, coverage_count, max_iters=200)
                coverage_count = rebuild_coverage(current_blocks)
                current_blocks = aggressive_deletion(current_blocks, coverage_count)
            
            return current_blocks, len(current_blocks)

    # Phase 1: Standard greedy with multiple seeds
    restart_count = 0
    while time.time() - start_time < time_limit - 30.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=False)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    # Phase 1b: Greedy Packing with Conflict Resolution
    if time.time() - start_time < time_limit - 20.0:
        for seed in range(8):
            if time.time() - start_time > time_limit - 20.0:
                break
            blocks, score = greedy_packing(seed, lambda_max=1, max_iters=400)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 1c: Greedy Packing with lambda_max=2 (allows some overlap, more blocks but denser)
    if time.time() - start_time < time_limit - 15.0:
        for seed in range(4):
            if time.time() - start_time > time_limit - 15.0:
                break
            blocks, score = greedy_packing(1000 + seed, lambda_max=2, max_iters=300)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 2: Column generation with dual-guided swaps
    if time.time() - start_time < time_limit - 12.0:
        for seed in range(6):
            if time.time() - start_time > time_limit - 12.0:
                break
            blocks, score = column_generation(seed, max_cg_iters=25)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 3: Randomized weighted greedy
    if time.time() - start_time < time_limit - 8.0:
        for seed in range(3):
            if time.time() - start_time > time_limit - 8.0:
                break
            blocks, score = run_greedy(2000 + seed, use_random_weights=True)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks
>>>>>>> REPLACE

