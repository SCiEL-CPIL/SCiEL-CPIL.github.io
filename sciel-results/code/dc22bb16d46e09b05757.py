    def local_search(code_indices, deadline):
        counts = np.zeros(N, dtype=np.int16)
        for c in code_indices:
            for j in ball_indices[c]:
                counts[j] += 1
        
        code_set = set(code_indices)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(code_indices)
            
            # Removal Pass
            for i in range(len(code_indices) - 1, -1, -1):
                if time.time() > deadline: return code_indices
                c = code_indices[i]
                is_redundant = True
                for j in ball_indices[c]:
                    if counts[j] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for j in ball_indices[c]:
                        counts[j] -= 1
                    code_indices.pop(i)
                    code_set.discard(c)
                    improved = True
            
            if not improved and len(code_indices) > 0:
                # Swap Pass
                for i in range(len(code_indices)):
                    if time.time() > deadline: return code_indices
                    c = code_indices[i]
                    
                    uncovered_indices = []
                    for j in ball_indices[c]:
                        if counts[j] == 1:
                            uncovered_indices.append(j)
                    
                    if not uncovered_indices:
                        continue
                    
                    # Try to find a swap
                    # To speed up, we can restrict candidates to those that cover at least one uncovered index?
                    # Or just scan all.
                    # Let's scan all, but break early if found.
                    # Randomize the scan order? 
                    # Shuffling N=2187 list is fast (~0.001s).
                    candidates = list(range(N))
                    random.shuffle(candidates)
                    
                    swap_found = False
                    for new_c in candidates:
                        if new_c in code_set:
                            continue
                        
                        # Check coverage
                        ok = True
                        for j in uncovered_indices:
                            if j not in ball_sets[new_c]:
                                ok = False
                                break
                        
                        if ok:
                            # Swap
                            for j in ball_indices[c]:
                                counts[j] -= 1
                            for j in ball_indices[new_c]:
                                counts[j] += 1
                            
                            code_indices[i] = new_c
                            code_set.discard(c)
                            code_set.add(new_c)
                            
                            improved = True
                            swap_found = True
                            break
                            
        return code_indices
