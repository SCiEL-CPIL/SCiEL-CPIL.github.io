<<<<<<< SEARCH
    # --- Final Polish Pass ---
    # Use the global_freq to bias a final high-precision local search on the best solution found.
    # We restrict the swap/add candidates to the top 5000 most frequent blocks to speed up the search.
    if _elapsed() < _DEADLINE - 10 and best_blocks is not None:
        # Rebuild cnt for best_blocks
        cnt_best = [0] * n_t
        for b in best_blocks:
            for idx in coverage[b]:
                cnt_best[idx] += 1
        
        # Check validity
        if all(c > 0 for c in cnt_best):
            # Identify top candidates
            sorted_blocks = sorted(global_freq.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [b for b, _ in sorted_blocks[:5000]]
            if not top_candidates:
                top_candidates = list(range(n_k))
            
            # Run local search again, but with a restricted candidate pool for swaps/adds?
            # The current _local_search iterates all n_k. 
            # To implement restricted pool, we'd need to modify _local_search or create a wrapper.
            # Given time constraints, we just run _local_search again with the bias, 
            # relying on the bias to guide the *greedy* if we were to rebuild, 
            # but LS uses random swaps. 
            # Instead, let's do a "Simulated Annealing" style final polish:
            # Try to remove, then if stuck, try swapping with a high-frequency block.
            
            rng_polish = random.Random(42)
            improved = True
            while improved and _elapsed() < _DEADLINE - 2:
                improved = False
                # Removal
                i = 0
                while i < len(best_blocks) and _elapsed() < _DEADLINE - 2:
                    b = best_blocks[i]
                    removable = True
                    for idx in coverage[b]:
                        if cnt_best[idx] == 1:
                            removable = False
                            break
                    if removable:
                        for idx in coverage[b]:
                            cnt_best[idx] -= 1
                        best_blocks.pop(i)
                        improved = True
                        continue
                    i += 1
                
                # Targeted Swap with High-Freq blocks
                if not improved:
                    # Pick a random block from best_blocks
                    if best_blocks:
                        idx_old = rng_polish.randrange(len(best_blocks))
                        old_b = best_blocks[idx_old]
                        
                        # Pick a random candidate from top_candidates not in best_blocks
                        in_set = set(best_blocks)
                        candidates = [c for c in top_candidates if c not in in_set]
                        if candidates:
                            new_b = candidates[rng_polish.randrange(len(candidates))]
                            
                            # Tentative swap
                            for idx in coverage[old_b]:
                                cnt_best[idx] -= 1
                            for idx in coverage[new_b]:
                                cnt_best[idx] += 1
                            best_blocks[idx_old] = new_b
                            
                            # Check if any removal is now possible
                            removed = False
                            for idx in range(len(best_blocks)):
                                b = best_blocks[idx]
                                removable = True
                                for tidx in coverage[b]:
                                    if cnt_best[tidx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tidx in coverage[b]:
                                        cnt_best[tidx] -= 1
                                    best_blocks.pop(idx)
                                    removed = True
                                    improved = True
                                    break
                            
                            if not removed:
                                # Revert
                                best_blocks[idx_old] = old_b
                                for idx in coverage[new_b]:
                                    cnt_best[idx] -= 1
                                for idx in coverage[old_b]:
                                    cnt_best[idx] += 1
=======
    # --- Final Polish Pass ---
    # Use the global_freq to bias a final high-precision local search on the best solution found.
    # We restrict the swap/add candidates to the top 5000 most frequent blocks to speed up the search.
    if _elapsed() < _DEADLINE - 10 and best_blocks is not None:
        # Rebuild cnt for best_blocks
        cnt_best = [0] * n_t
        for b in best_blocks:
            for idx in coverage[b]:
                cnt_best[idx] += 1
        
        # Check validity
        if all(c > 0 for c in cnt_best):
            # Identify top candidates
            sorted_blocks = sorted(global_freq.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [b for b, _ in sorted_blocks[:5000]]
            if not top_candidates:
                top_candidates = list(range(n_k))
            
            # Run local search again, but with a restricted candidate pool for swaps/adds?
            # The current _local_search iterates all n_k. 
            # To implement restricted pool, we'd need to modify _local_search or create a wrapper.
            # Given time constraints, we just run _local_search again with the bias, 
            # relying on the bias to guide the *greedy* if we were to rebuild, 
            # but LS uses random swaps. 
            # Instead, let's do a "Simulated Annealing" style final polish:
            # Try to remove, then if stuck, try swapping with a high-frequency block.
            
            rng_polish = random.Random(42)
            improved = True
            while improved and _elapsed() < _DEADLINE - 2:
                improved = False
                # Removal
                i = 0
                while i < len(best_blocks) and _elapsed() < _DEADLINE - 2:
                    b = best_blocks[i]
                    removable = True
                    for idx in coverage[b]:
                        if cnt_best[idx] == 1:
                            removable = False
                            break
                    if removable:
                        for idx in coverage[b]:
                            cnt_best[idx] -= 1
                        best_blocks.pop(i)
                        improved = True
                        continue
                    i += 1
                
                # Targeted Swap with High-Freq blocks
                if not improved:
                    # Pick a random block from best_blocks
                    if best_blocks:
                        idx_old = rng_polish.randrange(len(best_blocks))
                        old_b = best_blocks[idx_old]
                        
                        # Pick a random candidate from top_candidates not in best_blocks
                        in_set = set(best_blocks)
                        candidates = [c for c in top_candidates if c not in in_set]
                        if candidates:
                            new_b = candidates[rng_polish.randrange(len(candidates))]
                            
                            # Tentative swap
                            for idx in coverage[old_b]:
                                cnt_best[idx] -= 1
                            for idx in coverage[new_b]:
                                cnt_best[idx] += 1
                            best_blocks[idx_old] = new_b
                            
                            # Check if any removal is now possible
                            removed = False
                            for idx in range(len(best_blocks)):
                                b = best_blocks[idx]
                                removable = True
                                for tidx in coverage[b]:
                                    if cnt_best[tidx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tidx in coverage[b]:
                                        cnt_best[tidx] -= 1
                                    best_blocks.pop(idx)
                                    removed = True
                                    improved = True
                                    break
                            
                            if not removed:
                                # Revert
                                best_blocks[idx_old] = old_b
                                for idx in coverage[new_b]:
                                    cnt_best[idx] -= 1
                                for idx in coverage[old_b]:
                                    cnt_best[idx] += 1
                
                # 2-opt swap in polish
                if not improved and len(best_blocks) >= 2:
                    i1 = rng_polish.randrange(len(best_blocks))
                    i2 = rng_polish.randrange(len(best_blocks) - 1)
                    if i2 >= i1:
                        i2 += 1
                    old1 = best_blocks[i1]
                    old2 = best_blocks[i2]
                    
                    in_set = set(best_blocks)
                    cands = [c for c in top_candidates if c not in in_set]
                    if len(cands) >= 2:
                        j1 = cands[rng_polish.randrange(len(cands))]
                        j2 = cands[rng_polish.randrange(len(cands))]
                        if j1 != j2:
                            # Tentatively swap both
                            for idx in coverage[old1]:
                                cnt_best[idx] -= 1
                            for idx in coverage[old2]:
                                cnt_best[idx] -= 1
                            for idx in coverage[j1]:
                                cnt_best[idx] += 1
                            for idx in coverage[j2]:
                                cnt_best[idx] += 1
                            best_blocks[i1] = j1
                            best_blocks[i2] = j2
                            
                            # Try to remove blocks
                            removed_count = 0
                            idx = 0
                            while idx < len(best_blocks):
                                b = best_blocks[idx]
                                removable = True
                                for tidx in coverage[b]:
                                    if cnt_best[tidx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tidx in coverage[b]:
                                        cnt_best[tidx] -= 1
                                    best_blocks.pop(idx)
                                    removed_count += 1
                                    improved = True
                                    continue
                                idx += 1
                            
                            if removed_count < 2:
                                # Revert
                                best_blocks[i1] = old1
                                best_blocks[i2] = old2
                                for idx in coverage[j1]:
                                    cnt_best[idx] -= 1
                                for idx in coverage[j2]:
                                    cnt_best[idx] -= 1
                                for idx in coverage[old1]:
                                    cnt_best[idx] += 1
                                for idx in coverage[old2]:
                                    cnt_best[idx] += 1
>>>>>>> REPLACE
