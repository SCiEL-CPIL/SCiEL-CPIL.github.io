<<<<<<< SEARCH
    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find replacement using reverse_cover
                    # Candidates that cover all critical words
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code
=======
    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    return code
                
                c_remove = code[i]
                neighbors_remove = neighbors_arr[c_remove, :]
                counts[neighbors_remove] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_remove] += 1
                    
                    # Find critical words (covered only by c_remove)
                    critical_mask = (counts[neighbors_remove] == 1)
                    if not np.any(critical_mask):
                        continue
                        
                    critical_words = neighbors_remove[critical_mask]
                    
                    # Find replacement using reverse_cover
                    # Candidates that cover all critical words
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    # Heuristic: pick candidate that covers the most critical words
                    # Since all candidates cover all critical words by definition, 
                    # we can pick one that covers the most *other* uncovered or low-count words?
                    # Actually, just random is okay, but let's try picking the one with smallest index or similar for determinism? 
                    # No, random is better for diversity.
                    
                    best_w = random.choice(list(candidate_set))
                    neighbors_add = neighbors_arr[best_w, :]
                    code[i] = best_w
                    counts[neighbors_remove] -= 1
                    counts[neighbors_add] += 1
                    improved = True
                else:
                    code.pop(i)
                    improved = True
        
        return code
>>>>>>> REPLACE

Now, update the main loop to allow for a final polish and potentially more iterations if the code is smaller.

