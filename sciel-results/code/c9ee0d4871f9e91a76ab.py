<<<<<<< SEARCH
    # 5. Advanced Local Search: 1-for-1, 2-for-1, and 2-for-2 swaps
    if best_solution is not None and time.time() < deadline - 8:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        # Build index: for each subset, list of blocks containing it (for fast lookup)
        # Already have blocks_of_subset
        
        improved = True
        max_iters = 20
        iter_count = 0
        while improved and time.time() < deadline - 8 and iter_count < max_iters:
            improved = False
            iter_count += 1
            new_solution = best_solution.copy()
            solution_set = set(best_solution)
            
            # Try 1-for-1 replacement: remove block i, find if another block can cover all its unique subsets
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 8:
                    break
                subs_b = subsets_of_block[b_idx]
                
                # Find subsets that would become uncovered if we remove b_idx
                unique_subs = [s for s in subs_b if cover_count[s] == 1]
                
                if not unique_subs:
                    # Can remove entirely
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    solution_set.discard(b_idx)
                    improved = True
                    # Recompute cover counts
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx2 in new_solution:
                        for s_idx in subsets_of_block[b_idx2]:
                            cover_count[s_idx] += 1
                    break
                else:
                    # Try to find a replacement block that covers all unique_subs
                    # Use the most constrained subset to guide search
                    # Sort unique_subs by number of covering blocks (ascending)
                    unique_subs_sorted = sorted(unique_subs, key=lambda s: len(blocks_of_subset[s]))
                    
                    best_replacement = -1
                    for b_new in range(n_k):
                        if b_new in solution_set:
                            continue
                        # Check if b_new covers all unique_subs
                        # Quick check: does b_new contain the most constrained subset?
                        if unique_subs_sorted:
                            most_constrained = unique_subs_sorted[0]
                            if most_constrained not in subsets_of_block[b_new]:
                                continue
                        new_subs_set = set(subsets_of_block[b_new])
                        if all(s in new_subs_set for s in unique_subs):
                            best_replacement = b_new
                            break
                    
                    if best_replacement != -1:
                        # Remove b_idx, add best_replacement
                        cover_count[subs_b] -= 1
                        new_solution[i] = best_replacement
                        solution_set.discard(b_idx)
                        solution_set.add(best_replacement)
                        for s_idx in subsets_of_block[best_replacement]:
                            cover_count[s_idx] += 1
                        improved = True
                        break
            
            # If no 1-for-1 found, try 2-for-1: remove two blocks, add one that covers all their unique subsets
            if not improved and len(best_solution) >= 2:
                for i in range(len(best_solution)):
                    if time.time() > deadline - 8:
                        break
                    b_i = best_solution[i]
                    subs_i = subsets_of_block[b_i]
                    unique_i = [s for s in subs_i if cover_count[s] == 1]
                    
                    for j in range(i + 1, len(best_solution)):
                        b_j = best_solution[j]
                        subs_j = subsets_of_block[b_j]
                        unique_j = [s for s in subs_j if cover_count[s] == 1]
                        
                        # Combined unique subsets
                        combined_unique = set(unique_i) | set(unique_j)
                        if not combined_unique:
                            continue
                        
                        # Find a block that covers all combined_unique
                        best_replacement = -1
                        for b_new in range(n_k):
                            if b_new in solution_set:
                                continue
                            new_subs_set = set(subsets_of_block[b_new])
                            if all(s in new_subs_set for s in combined_unique):
                                best_replacement = b_new
                                break
                        
                        if best_replacement != -1:
                            # Remove b_i and b_j, add best_replacement
                            cover_count[subs_i] -= 1
                            cover_count[subs_j] -= 1
                            # Remove both from new_solution
                            new_solution.pop(j)  # Remove j first (higher index)
                            new_solution.pop(i)  # Now remove i
                            solution_set.discard(b_i)
                            solution_set.discard(b_j)
                            new_solution.append(best_replacement)
                            solution_set.add(best_replacement)
                            for s_idx in subsets_of_block[best_replacement]:
                                cover_count[s_idx] += 1
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)
=======
    # 5. Advanced Local Search: efficient 1-for-1, 2-for-1, and 2-for-2 swaps
    if best_solution is not None and time.time() < deadline - 8:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        # Precompute for each block: set of t-subsets it covers (for fast membership)
        block_subsets_set = [set(subsets_of_block[j]) for j in range(len(best_solution))]
        
        improved = True
        max_iters = 30
        iter_count = 0
        while improved and time.time() < deadline - 8 and iter_count < max_iters:
            improved = False
            iter_count += 1
            new_solution = best_solution.copy()
            solution_set = set(best_solution)
            
            # Try 1-for-1 replacement: remove block i, find if another block can cover all its unique subsets
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 8:
                    break
                subs_b = subsets_of_block[b_idx]
                
                # Find subsets that would become uncovered if we remove b_idx
                unique_subs = [s for s in subs_b if cover_count[s] == 1]
                
                if not unique_subs:
                    # Can remove entirely
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    solution_set.discard(b_idx)
                    improved = True
                    # Recompute cover counts
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx2 in new_solution:
                        for s_idx in subsets_of_block[b_idx2]:
                            cover_count[s_idx] += 1
                    break
                else:
                    # Try to find a replacement block that covers all unique_subs
                    # Use the most constrained subset to guide search
                    unique_subs_sorted = sorted(unique_subs, key=lambda s: len(blocks_of_subset[s]))
                    most_constrained = unique_subs_sorted[0]
                    
                    # Only check blocks that contain the most constrained subset
                    best_replacement = -1
                    for b_new in blocks_of_subset[most_constrained]:
                        if b_new in solution_set:
                            continue
                        # Check if b_new covers all unique_subs
                        if all(s in block_subsets_set[b_new] for s in unique_subs):
                            best_replacement = b_new
                            break
                    
                    if best_replacement != -1:
                        # Remove b_idx, add best_replacement
                        cover_count[subs_b] -= 1
                        new_solution[i] = best_replacement
                        solution_set.discard(b_idx)
                        solution_set.add(best_replacement)
                        for s_idx in subsets_of_block[best_replacement]:
                            cover_count[s_idx] += 1
                        improved = True
                        break
            
            # If no 1-for-1 found, try 2-for-1: remove two blocks, add one that covers all their unique subsets
            if not improved and len(best_solution) >= 2:
                for i in range(len(best_solution)):
                    if time.time() > deadline - 8:
                        break
                    b_i = best_solution[i]
                    subs_i = subsets_of_block[b_i]
                    unique_i = [s for s in subs_i if cover_count[s] == 1]
                    
                    for j in range(i + 1, len(best_solution)):
                        b_j = best_solution[j]
                        subs_j = subsets_of_block[b_j]
                        unique_j = [s for s in subs_j if cover_count[s] == 1]
                        
                        # Combined unique subsets
                        combined_unique = set(unique_i) | set(unique_j)
                        if not combined_unique:
                            continue
                        
                        # Find a block that covers all combined_unique
                        # Use the most constrained subset from combined_unique
                        combined_sorted = sorted(combined_unique, key=lambda s: len(blocks_of_subset[s]))
                        most_constrained_c = combined_sorted[0]
                        
                        best_replacement = -1
                        for b_new in blocks_of_subset[most_constrained_c]:
                            if b_new in solution_set:
                                continue
                            if all(s in block_subsets_set[b_new] for s in combined_unique):
                                best_replacement = b_new
                                break
                        
                        if best_replacement != -1:
                            # Remove b_i and b_j, add best_replacement
                            cover_count[subs_i] -= 1
                            cover_count[subs_j] -= 1
                            new_solution.pop(j)
                            new_solution.pop(i)
                            solution_set.discard(b_i)
                            solution_set.discard(b_j)
                            new_solution.append(best_replacement)
                            solution_set.add(best_replacement)
                            for s_idx in subsets_of_block[best_replacement]:
                                cover_count[s_idx] += 1
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)
                # Recompute block_subsets_set for new solution
                block_subsets_set = [set(subsets_of_block[j]) for j in best_solution]
>>>>>>> REPLACE

