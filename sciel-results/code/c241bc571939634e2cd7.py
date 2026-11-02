<<<<<<< SEARCH
    # Final local search polish on best_set
    selected = best_set.copy()
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        cover_count[balls[c]] += 1
    
    # Phase 1: Remove redundant
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Phase 2: Swap - multiple passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 1.0:
        improved = False
        pass_count += 1
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                improved = True
                continue
            
            critical_mask = (cover_count[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                cover_count[balls[c]] -= 1
                improved = True
                continue

            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= ball_membership[:, cw]
            
            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]
            
            if len(valid_candidates) > 0:
                # Try multiple candidates if we have time
                best_c_prime = None
                for _ in range(min(3, len(valid_candidates))):
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    # Check if this swap is beneficial (reduces max cover count)
                    old_max = np.max(cover_count[balls[c_prime]])
                    new_max = np.max(cover_count[balls[c]])
                    if old_max >= new_max:
                        best_c_prime = c_prime
                        break
                if best_c_prime is None:
                    best_c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                
                cover_count[balls[c]] -= 1
                cover_count[balls[best_c_prime]] += 1
                new_selected.append(best_c_prime)
                improved = True
            else:
                new_selected.append(c)
        
        selected = new_selected
    
    # Phase 3: Another round of redundancy removal
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    if len(selected) < best_size:
        best_size = len(selected)
        best_set = selected.copy()
=======
    # Final local search polish on best_set
    selected = best_set.copy()
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        cover_count[balls[c]] += 1
    
    # Phase 1: Remove redundant
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Phase 2: Swap - multiple passes using bit-parallel masks
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 2.0:
        improved = False
        pass_count += 1
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                improved = True
                continue
            
            critical_mask = (cover_count[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                cover_count[balls[c]] -= 1
                improved = True
                continue

            # Use bit-parallel AND to find candidates covering all critical words
            candidate_bits = (1 << total) - 1
            for cw in critical_words:
                candidate_bits &= ball_mask[cw]
            
            if candidate_bits == 0:
                new_selected.append(c)
                continue
            
            # Extract valid candidate indices
            valid_candidates = []
            bits = candidate_bits
            while bits:
                lsb = bits & (-bits)
                pos = lsb.bit_length() - 1
                valid_candidates.append(pos)
                bits ^= lsb
            
            valid_candidates = [v for v in valid_candidates if v != c]
            
            if len(valid_candidates) > 0:
                # Try multiple candidates if we have time
                best_c_prime = None
                for _ in range(min(3, len(valid_candidates))):
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    # Check if this swap is beneficial (reduces max cover count)
                    old_max = np.max(cover_count[balls[c_prime]])
                    new_max = np.max(cover_count[balls[c]])
                    if old_max >= new_max:
                        best_c_prime = c_prime
                        break
                if best_c_prime is None:
                    best_c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                
                cover_count[balls[c]] -= 1
                cover_count[balls[best_c_prime]] += 1
                new_selected.append(best_c_prime)
                improved = True
            else:
                new_selected.append(c)
        
        selected = new_selected
    
    # Phase 3: Another round of redundancy removal
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    if len(selected) < best_size:
        best_size = len(selected)
        best_set = selected.copy()
>>>>>>> REPLACE

