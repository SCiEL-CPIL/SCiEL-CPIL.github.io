<<<<<<< SEARCH
        current = list(initial_indices)
        if not current:
            return []
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 3
=======
        current = list(initial_indices)
        if not current:
            return []
        
        # ILS perturbation: randomly remove a few blocks and repair
        if perturbation and len(current) > 5:
            rng_pert = random.Random()
            n_remove = max(2, len(current) // 15)
            to_remove = rng_pert.sample(range(len(current)), n_remove)
            to_remove.sort(reverse=True)
            for i in to_remove:
                b_idx = current[i]
                for ts_idx in block_to_tsets[b_idx]:
                    cover_counts[ts_idx] -= 1
                current.pop(i)
            # Greedy repair for uncovered t-sets
            uncovered_ts = [i for i in range(num_tsets) if cover_counts[i] == 0]
            if uncovered_ts:
                # Find blocks that cover many uncovered t-sets
                for b_idx in range(num_blocks):
                    if b_idx in current:
                        continue
                    gain = 0
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 0:
                            gain += 1
                    if gain > 0:
                        current.append(b_idx)
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] += 1
                        uncovered_ts = [i for i in range(num_tsets) if cover_counts[i] == 0]
                        if not uncovered_ts:
                            break
        
        # Calculate initial cover counts (recompute for safety after perturbation)
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 5
>>>>>>> REPLACE

