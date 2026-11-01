<<<<<<< SEARCH
    def local_search(code):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pass 1: Removals
            for c in list(code_set):
                if cover_count[neighbors_arr[c]].min() > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    improved = True
            
            if improved:
                continue
                
            # Pass 2: Swaps
            # Try to find a valid swap to perturb the state
            # We limit attempts to avoid wasting time on dead ends
            for _ in range(50):
                if not code_set:
                    break
                c = random.choice(list(code_set))
                # Get critical words (covered only by c)
                crit = [w for w in neighbors[c] if cover_count[w] == 1]
                if not crit:
                    # Should have been removed, but maybe due to order?
                    # Force removal
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    break
                    
                # Try random x
                x = random.randint(0, total_words - 1)
                if x in code_set:
                    continue
                    
                # Check if x covers all critical words of c
                valid = True
                for w in crit:
                    if not (cover_bits[x] >> w) & 1:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    cover_count[neighbors_arr[c]] -= 1
                    cover_count[neighbors_arr[x]] += 1
                    code_set.remove(c)
                    code_set.add(x)
                    improved = True
                    break
                    
        return list(code_set)
=======
    def local_search(code, max_iterations=10000):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        improved = True
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Pass 1: Aggressive removals - remove all redundant codewords
            for c in list(code_set):
                if cover_count[neighbors_arr[c]].min() > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    improved = True
            
            if improved:
                continue
                
            # Pass 2: Try to find a beneficial swap
            # First, identify codewords with few critical words (easier to replace)
            candidates_with_crit = []
            for c in code_set:
                crit_mask = np.zeros(total_words, dtype=bool)
                crit_mask[neighbors[c]] = True
                crit_words = np.where(crit_mask & (cover_count == 1))[0]
                if len(crit_words) > 0:
                    candidates_with_crit.append((len(crit_words), c, crit_words))
            
            if not candidates_with_crit:
                # All codewords have no critical words? Should have been removed
                break
            
            # Sort by number of critical words (try easiest replacements first)
            candidates_with_crit.sort(key=lambda x: x[0])
            
            # Try swaps for a limited number of candidates
            swap_found = False
            for _, c, crit_words in candidates_with_crit[:10]:
                if swap_found or time.time() >= deadline:
                    break
                
                # Try multiple random replacements
                for _ in range(20):
                    x = random.randint(0, total_words - 1)
                    if x in code_set:
                        continue
                    
                    # Check if x covers all critical words of c
                    valid = True
                    for w in crit_words:
                        if not (cover_bits[x] >> w) & 1:
                            valid = False
                            break
                    
                    if valid:
                        # Check if this swap doesn't uncover any words
                        # (it shouldn't if x covers all critical words, but let's be safe)
                        new_cover = cover_count.copy()
                        new_cover[neighbors_arr[c]] -= 1
                        new_cover[neighbors_arr[x]] += 1
                        if new_cover.min() >= 1:
                            # Perform swap
                            cover_count = new_cover
                            code_set.remove(c)
                            code_set.add(x)
                            improved = True
                            swap_found = True
                            break
        
        return list(code_set)
>>>>>>> REPLACE
