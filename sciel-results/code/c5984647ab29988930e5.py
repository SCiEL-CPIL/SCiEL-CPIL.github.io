<<<<<<< SEARCH
    # Local Search Optimizer
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        max_iters = 5000
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                pairs_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs_to_try)
                
                for i, j in pairs_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                    
                    needed_triplets = []
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        # Both redundant
                        b_remove = blocks.pop(j)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            # Swap
                            b_remove = blocks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                                
                            blocks.append(new_block)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 1-for-1
            if not improved:
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1):
                                continue
                            new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                            if all(tid in new_ids for tid in needed_triplets):
                                b_remove = blocks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True
                                break
                                
            if not improved:
                break
                
        return blocks
=======
    # Local Search Optimizer
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        max_iters = 20000
        no_improve_count = 0
        stall = 0
        max_stall = 3000
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if stall > max_stall:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                pairs_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs_to_try)
                
                for i, j in pairs_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                    
                    needed_triplets = []
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        # Both redundant
                        b_remove = blocks.pop(j)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        # FIXED: check only needed_triplets, not all_involved
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            # Swap
                            b_remove = blocks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                                
                            blocks.append(new_block)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 3-for-2
            if not improved:
                triples_to_try = [(a, b, c) for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
                random.shuffle(triples_to_try)
                
                for a, b, c in triples_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    b_a = blocks[a]
                    b_b = blocks[b]
                    b_c = blocks[c]
                    
                    ids_a = set(triplet_to_id[triplet] for triplet in itertools.combinations(b_a, t))
                    ids_b = set(triplet_to_id[triplet] for triplet in itertools.combinations(b_b, t))
                    ids_c = set(triplet_to_id[triplet] for triplet in itertools.combinations(b_c, t))
                    
                    all_involved = ids_a | ids_b | ids_c
                    needed_triplets = [tid for tid in all_involved if counts[tid] == 1]
                    
                    if not needed_triplets:
                        # All three redundant
                        for idx in [c, b, a]:
                            b_remove = blocks.pop(idx)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                    
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b_a) or set(new_block) == set(b_b) or set(new_block) == set(b_c):
                            continue
                            
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            # Swap: remove 3, add 2 (one new block + keep one existing)
                            # Actually we need to add 2 blocks total to replace 3
                            # So we add 1 new block and keep 1 of the original 3
                            # Find which original block to keep
                            kept_idx = None
                            for idx in [a, b, c]:
                                b_keep = blocks[idx]
                                keep_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(b_keep, t))
                                if all(tid in keep_ids for tid in needed_triplets):
                                    kept_idx = idx
                                    break
                            
                            if kept_idx is None:
                                # Need 2 new blocks
                                # Try to find 2 blocks that together cover all needed
                                # For simplicity, skip this case for now
                                continue
                            
                            # Remove the 2 blocks we're not keeping
                            remove_indices = sorted([x for x in [a, b, c] if x != kept_idx], reverse=True)
                            for idx in remove_indices:
                                b_remove = blocks.pop(idx)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                            
                            blocks.append(new_block)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                            
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 4. Try 1-for-1
            if not improved:
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1):
                                continue
                            new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                            if all(tid in new_ids for tid in needed_triplets):
                                b_remove = blocks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True
                                break
                                
            if not improved:
                stall += 1
            else:
                stall = 0
                
        return blocks
>>>>>>> REPLACE
