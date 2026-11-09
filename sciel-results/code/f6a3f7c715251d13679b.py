<<<<<<< SEARCH
def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Simulated annealing parameters
    temperature = 3.0
    cooling_rate = 0.995
    min_temperature = 0.05
    
    n_blocks_total = len(triple_to_blocks[0]) if triple_to_blocks else 0
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
        
        if not current_blocks:
            break
            
        # Phase 2: Try moves with simulated annealing
        move_type = rng.random()
        
        if move_type < 0.7:
            # 1-exchange: remove one block, add one block
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            uncovered_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    uncovered_triples.append(idx)
                    
            if not uncovered_triples:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue

            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                # Try up to 3 candidates
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # Compute quality: penalize unique coverage, reward redundancy
                    # Quality = sum over triples of (1 if count >= 2 else 0) - lambda * (count == 1)
                    # Simpler: quality = sum of min(count, 2) for all triples
                    # But we need a faster metric. Use: number of triples with count == 1 (bad) 
                    # and number with count >= 2 (good).
                    
                    # Old quality: count of triples with count >= 2
                    old_redundant = 0
                    for idx in block_triple_indices[b_out]:
                        if triple_counts[idx] >= 2:
                            old_redundant += 1
                    # Also need to account for triples NOT in b_out that might be affected? No, b_out only affects its own triples.
                    
                    # New quality: simulate
                    new_redundant = 0
                    for idx in block_triple_indices[b_out]:
                        new_c = triple_counts[idx] - 1
                        if new_c >= 2:
                            new_redundant += 1
                    for idx in block_triple_indices[best_candidate]:
                        if idx in block_triple_indices[b_out]:
                            # Already counted above
                            pass
                        else:
                            new_c = triple_counts[idx] + 1
                            if new_c >= 2:
                                new_redundant += 1
                    
                    # Also count triples in b_out that are NOT in best_candidate
                    # And triples in best_candidate that are NOT in b_out
                    # The above logic is flawed. Let's just compute the delta properly.
                    
                    # delta_quality = (new_redundant - old_redundant)
                    # But we also care about not creating new unique coverages.
                    # Let's use a penalty for triples that become unique (count 1) from being redundant (count >= 2)
                    # or from being unique to 0 (which is invalid, so we avoid those).
                    
                    # Better metric: sum of (1 if count >= 2 else 0) - 0.5 * (1 if count == 1 else 0)
                    # This rewards redundancy and penalizes unique coverage.
                    
                    old_metric = 0
                    for idx in block_triple_indices[b_out]:
                        c = triple_counts[idx]
                        if c >= 2:
                            old_metric += 1
                        elif c == 1:
                            old_metric -= 0.5
                    for idx in block_triple_indices[best_candidate]:
                        if idx not in block_triple_indices[b_out]:
                            c = triple_counts[idx]
                            if c >= 2:
                                old_metric += 1
                            elif c == 1:
                                old_metric -= 0.5
                            
                    new_metric = 0
                    # We need to compute the new counts for affected triples
                    # Affected triples are those in b_out or best_candidate
                    affected = set(block_triple_indices[b_out]) | set(block_triple_indices[best_candidate])
                    for idx in affected:
                        c = triple_counts[idx]
                        if idx in block_triple_indices[b_out]:
                            c -= 1
                        if idx in block_triple_indices[best_candidate]:
                            c += 1
                        if c >= 2:
                            new_metric += 1
                        elif c == 1:
                            new_metric -= 0.8 # Heavier penalty for unique coverage
                    
                    delta_quality = new_metric - old_metric
                    
                    if delta_quality >= 0 or rng.random() < np.exp(delta_quality / temperature):
                        del current_blocks[pos]
                        for idx in block_triple_indices[b_out]:
                            triple_counts[idx] -= 1
                        
                        current_blocks.append(best_candidate)
                        for idx in block_triple_indices[best_candidate]:
                            triple_counts[idx] += 1
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
        elif move_type < 0.9:
            # 2-exchange: remove two blocks, add one block
            if len(current_blocks) < 2:
                continue
                
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2:
                continue
                
            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]
            
            # Temporarily remove both
            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1
            
            # Find uncovered triples
            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                # Both are redundant
                # Remove the one with lower index first to avoid issues
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                continue
            
            # Find a block that covers all uncovered triples
            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # This reduces block count by 1, so always accept
                    # Remove both blocks
                    if pos1 > pos2:
                        pos1, pos2 = pos2, pos1
                        b_out1, b_out2 = b_out2, b_out1
                    del current_blocks[pos1]
                    for idx in block_triple_indices[b_out1]:
                        triple_counts[idx] -= 1
                    del current_blocks[pos2 - 1]
                    for idx in block_triple_indices[b_out2]:
                        triple_counts[idx] -= 1
                    
                    current_blocks.append(best_candidate)
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1
                    
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
    
    return best_blocks
=======
def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, n_triples, deadline):
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Simulated annealing parameters
    temperature = 3.0
    cooling_rate = 0.998
    min_temperature = 0.01
    
    # Track which blocks are currently in use for O(1) lookup
    in_use = [False] * len(triple_to_blocks[0]) if triple_to_blocks else []
    for b in current_blocks:
        in_use[b] = True
    
    def compute_metric(counts, affected_triples):
        """Compute quality metric for affected triples only."""
        metric = 0
        for idx in affected_triples:
            c = counts[idx]
            if c >= 2:
                metric += 1.0
            elif c == 1:
                metric -= 0.8
        return metric
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks (fast pass)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    in_use[b_out] = False
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
        
        if not current_blocks:
            break
            
        # Phase 2: Try moves
        move_type = rng.random()
        
        if move_type < 0.65:
            # 1-exchange: remove one block, add one block
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            uncovered_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    uncovered_triples.append(idx)
                    
            if not uncovered_triples:
                del current_blocks[pos]
                in_use[b_out] = False
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue

            # Find candidates that cover all uncovered triples
            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = list(triple_to_blocks[min_triple])
            
            # Filter to only those covering all uncovered triples and not already in use
            filtered_candidates = []
            for b_in in candidates:
                if in_use[b_in]:
                    continue
                valid = True
                for tri in uncovered_triples:
                    if tri != min_triple:
                        # Check if b_in covers tri
                        # We need to check if tri is in block_triple_indices[b_in]
                        # This is O(k^t) in worst case, but we can precompute or use a set
                        # For efficiency, we'll check by iterating
                        if tri not in block_triple_indices[b_in]:
                            valid = False
                            break
                if valid:
                    filtered_candidates.append(b_in)
            
            if filtered_candidates:
                # Score candidates by how many additional triples they cover
                scored_candidates = []
                for b_in in filtered_candidates:
                    extra_gain = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            extra_gain += 1
                    scored_candidates.append((extra_gain, b_in))
                
                scored_candidates.sort(key=lambda x: -x[0])
                best_candidate = scored_candidates[0][1]
                
                # Compute delta quality
                affected = set(block_triple_indices[b_out]) | set(block_triple_indices[best_candidate])
                old_metric = compute_metric(triple_counts, affected)
                
                # Simulate new counts
                new_counts = {}
                for idx in affected:
                    c = triple_counts[idx]
                    if idx in block_triple_indices[b_out]:
                        c -= 1
                    if idx in block_triple_indices[best_candidate]:
                        c += 1
                    new_counts[idx] = c
                
                new_metric = 0
                for idx in affected:
                    c = new_counts[idx]
                    if c >= 2:
                        new_metric += 1.0
                    elif c == 1:
                        new_metric -= 0.8
                
                delta_quality = new_metric - old_metric
                
                if delta_quality >= 0 or rng.random() < np.exp(delta_quality / temperature):
                    del current_blocks[pos]
                    in_use[b_out] = False
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    
                    current_blocks.append(best_candidate)
                    in_use[best_candidate] = True
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
        elif move_type < 0.85:
            # 2-exchange: remove two blocks, add one block
            if len(current_blocks) < 2:
                continue
                
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2:
                continue
                
            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]
            
            # Temporarily remove both
            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1
            
            # Find uncovered triples
            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                # Both are redundant
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                in_use[b_out1] = False
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                in_use[b_out2] = False
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                continue
            
            # Find a block that covers all uncovered triples
            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = list(triple_to_blocks[min_triple])
            
            filtered_candidates = []
            for b_in in candidates:
                if in_use[b_in]:
                    continue
                valid = True
                for tri in uncovered:
                    if tri != min_triple:
                        if tri not in block_triple_indices[b_in]:
                            valid = False
                            break
                if valid:
                    filtered_candidates.append(b_in)
            
            if filtered_candidates:
                # Score candidates
                scored_candidates = []
                for b_in in filtered_candidates:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    scored_candidates.append((gain, b_in))
                
                scored_candidates.sort(key=lambda x: -x[0])
                best_candidate = scored_candidates[0][1]
                
                # This reduces block count by 1, so always accept
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                in_use[b_out1] = False
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                in_use[b_out2] = False
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                
                current_blocks.append(best_candidate)
                in_use[best_candidate] = True
                for idx in block_triple_indices[best_candidate]:
                    triple_counts[idx] += 1
                
                temperature *= cooling_rate
                if temperature < min_temperature:
                    temperature = min_temperature
    
    return best_blocks
>>>>>>> REPLACE

