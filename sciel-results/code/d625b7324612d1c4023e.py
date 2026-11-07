<<<<<<< SEARCH
def _local_search(mask, deadline):
    """
    Local search to improve mask size.
    Strategy: 
    1. Try adding any single missing element.
    2. Try 'remove one, add many' moves.
    3. Try 'remove two, add many' moves (2-exchange).
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    stuck_rounds = 0
    max_stuck = 3
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                stuck_rounds = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try 'remove one, add many'
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
        
        for v in elems:
            if time.time() >= deadline:
                break
            bit = 1 << (v - 1)
            reduced_mask = current_mask & ~bit
            
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << (add_v - 1)
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_mask(trial):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
        
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
            stuck_rounds = 0
            continue
        
        # Phase 3: Try 'remove two, add many' (2-exchange)
        found_better_2 = False
        best_candidate_2 = current_mask
        best_candidate_2_size = current_size
        
        if len(elems) >= 2:
            # Limit the number of pairs to try for speed
            max_pairs = min(len(elems) * (len(elems) - 1) // 2, 200)
            pair_count = 0
            for idx_i in range(len(elems)):
                if time.time() >= deadline or pair_count >= max_pairs:
                    break
                for idx_j in range(idx_i + 1, len(elems)):
                    if time.time() >= deadline or pair_count >= max_pairs:
                        break
                    pair_count += 1
                    a = elems[idx_i]
                    b = elems[idx_j]
                    reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    
                    candidate = reduced
                    for add_v in range(1, N + 1):
                        add_bit = 1 << (add_v - 1)
                        if candidate & add_bit:
                            continue
                        trial = candidate | add_bit
                        if _is_valid_mask(trial):
                            candidate = trial
                    
                    if candidate.bit_count() > best_candidate_2_size:
                        best_candidate_2 = candidate
                        best_candidate_2_size = candidate.bit_count()
                        found_better_2 = True
        
        if found_better_2:
            current_mask = best_candidate_2
            current_size = best_candidate_2_size
            stuck_rounds = 0
            continue
        
        # No improvement in any phase
        stuck_rounds += 1
        if stuck_rounds >= max_stuck:
            break
        
    return current_mask
=======
def _local_search(mask, deadline):
    """
    Local search to improve mask size.
    Strategy: 
    1. Try adding any single missing element.
    2. Try 'remove one, add many' moves.
    3. Use Bron-Kerbosch to find multiple compatible additions at once.
    4. Try 'remove two, add many' moves (2-exchange).
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    stuck_rounds = 0
    max_stuck = 3
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                stuck_rounds = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try 'remove one, add many'
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
        
        for v in elems:
            if time.time() >= deadline:
                break
            bit = 1 << (v - 1)
            reduced_mask = current_mask & ~bit
            
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << (add_v - 1)
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_mask(trial):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
        
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
            stuck_rounds = 0
            continue
        
        # Phase 3: Bron-Kerbosch to find multiple compatible additions
        if time.time() < deadline - 5:
            adj, missing = _build_compatibility_graph(current_mask)
            if len(missing) > 0:
                clique = _bron_kerbosch_max_clique(adj, missing, deadline - 5)
                if len(clique) > 1:
                    new_mask = current_mask
                    for idx in clique:
                        v = missing[idx]
                        new_mask |= (1 << (v - 1))
                    if new_mask.bit_count() > current_size:
                        current_mask = new_mask
                        current_size = current_mask.bit_count()
                        stuck_rounds = 0
                        continue
        
        # Phase 4: Try 'remove two, add many' (2-exchange)
        found_better_2 = False
        best_candidate_2 = current_mask
        best_candidate_2_size = current_size
        
        if len(elems) >= 2:
            # Limit the number of pairs to try for speed
            max_pairs = min(len(elems) * (len(elems) - 1) // 2, 200)
            pair_count = 0
            for idx_i in range(len(elems)):
                if time.time() >= deadline or pair_count >= max_pairs:
                    break
                for idx_j in range(idx_i + 1, len(elems)):
                    if time.time() >= deadline or pair_count >= max_pairs:
                        break
                    pair_count += 1
                    a = elems[idx_i]
                    b = elems[idx_j]
                    reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    
                    candidate = reduced
                    for add_v in range(1, N + 1):
                        add_bit = 1 << (add_v - 1)
                        if candidate & add_bit:
                            continue
                        trial = candidate | add_bit
                        if _is_valid_mask(trial):
                            candidate = trial
                    
                    if candidate.bit_count() > best_candidate_2_size:
                        best_candidate_2 = candidate
                        best_candidate_2_size = candidate.bit_count()
                        found_better_2 = True
        
        if found_better_2:
            current_mask = best_candidate_2
            current_size = best_candidate_2_size
            stuck_rounds = 0
            continue
        
        # No improvement in any phase
        stuck_rounds += 1
        if stuck_rounds >= max_stuck:
            break
        
    return current_mask
>>>>>>> REPLACE

