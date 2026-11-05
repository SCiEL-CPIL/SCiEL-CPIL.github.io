        # 2. Local Search Phase
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Directed Swap
                # Try to replace a codeword with another that covers its critical neighbors
                # and maximizes redundancy coverage.
                swap_made = False
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    
                    if len(critical_words) == 0:
                        # Should have been removed in Step A, but just in case
                        C.remove(c)
                        in_C[c] = False
                        cover_count[nb_c] -= 1 # Wait, we didn't decrement in Step A if we didn't enter the block?
                        # In Step A, we decrement, check, and increment if bad.
                        # If we are here, it means Step A didn't remove it.
                        # So critical_words must be non-empty?
                        # If critical_words is empty, Step A would have removed it.
                        # So this branch is likely unreachable.
                        continue
                    
                    # Find candidates that cover all critical words
                    if len(critical_words) == 1:
                        candidates = neighbors[critical_words[0]]
                    else:
                        mask0 = np.zeros(N, dtype=bool)
                        mask0[neighbors[critical_words[0]]] = True
                        for w in critical_words[1:]:
                            mask_w = np.zeros(N, dtype=bool)
                            mask_w[neighbors[w]] = True
                            mask0 &= mask_w
                            if not mask0.any():
                                break
                        if not mask0.any():
                            continue
                        candidates = np.where(mask0)[0]
                    
                    mask = ~in_C[candidates]
                    if not np.any(mask):
                        continue
                    
                    valid_candidates = candidates[mask]
                    
                    best_cp = -1
                    best_score = -1
                    
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        # Score: Number of neighbors of cp that are currently covered >= 2 times
                        # This indicates cp reinforces safe areas.
                        score = np.sum(cover_count[nb_cp] >= 2)
                        
                        if score > best_score:
                            best_score = score
                            best_cp = cp
                    
                    if best_cp != -1:
                        # Perform swap
                        C.remove(c)
                        C.append(best_cp)
                        in_C[c] = False
                        in_C[best_cp] = True
                        cover_count[nb_c] -= 1
                        cover_count[neighbors[best_cp]] += 1
                        improved = True
                        swap_made = True
                        break # Restart inner loop to check for new redundancies
                
                # If no swap made, try a random perturbation?
                # For now, just break to allow outer loop to try again with new greedy.
    