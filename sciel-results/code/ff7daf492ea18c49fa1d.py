<<<<<<< SEARCH
    # ---- Phase 2: local search -- add one, and if that blocks, repair by
    #              removing a small number of AP elements. ----
    ap_list = sorted(from_mask(init_mask))
    pool = sorted(set(range(1, n + 1)) - set(from_mask(best_mask)))

    repair_budget = 2
    progress = 0
    attempts = 0
    while time_left():
        # Try to add each candidate (in ascending order) to the current best.
        added_any = False
        for x in pool:
            if not time_left():
                break
            if (best_mask >> x) & 1:
                continue
            nm, ok = try_add(best_mask, x)
            if ok:
                best_mask = nm
                best_size = nm.bit_count()
                progress = 0
                added_any = True
                # Rebuild pool (x is now used).
                pool = [y for y in pool if y != x]
                continue
            # x was blocked.  Attempt a repair: remove up to repair_budget AP
            # elements, then try to add x.
            if repair_budget >= 1 and len(ap_list) > best_size - repair_budget + 1:
                # find a small subset of current elements to drop so x fits.
                # Simple heuristic: drop the smallest AP elements one by one.
                cur = best_mask
                drop_candidates = [e for e in ap_list if (cur >> e) & 1]
                drop_candidates.sort()
                for r in range(1, repair_budget + 1):
                    if not time_left():
                        break
                    for combo_size in range(1, r + 1):
                        if len(drop_candidates) < combo_size:
                            break
                        # try dropping the `combo_size` smallest still-present AP elems
                        subset = drop_candidates[:combo_size]
                        trial = cur
                        for e in subset:
                            trial = try_remove(trial, e)
                        nm2, ok2 = try_add(trial, x)
                        if ok2 and nm2.bit_count() > best_size:
                            best_mask = nm2
                            best_size = nm2.bit_count()
                            progress = 0
                            added_any = True
                            # drop x from pool
                            pool = [y for y in pool if y != x]
                            break
                        if ok2:
                            pass
                    if added_any:
                        break
        attempts += 1
        if not added_any and attempts % 200 == 0:
            # nothing improved; stop to respect time budget
            break
=======
    # ---- Phase 2: Enhanced Local Search ----
    # Moves:
    # 1. Add x
    # 2. Remove u, Add x (Swap, size 0, might enable further adds)
    # 3. Remove u, Add x, Add y (Size +1)
    
    cur_list = sorted(from_mask(best_mask))
    pool = sorted(set(range(1, n + 1)) - set(cur_list))
    
    improved = True
    while improved and time_left():
        improved = False
        
        # 1. Try to add any x from pool
        for x in pool:
            if not time_left():
                break
            if (best_mask >> x) & 1:
                continue
            nm, ok = try_add(best_mask, x)
            if ok:
                best_mask = nm
                best_size = nm.bit_count()
                improved = True
                pool = [y for y in pool if y != x]
                cur_list = sorted(from_mask(best_mask))
                break # Restart loop with new state
        
        if improved:
            continue
            
        # 2. Try Remove u, Add x, Add y
        # We want to increase size by 1.
        # Iterate over u in current set.
        # Iterate over x, y in pool (x < y).
        # Check if (S - {u}) + {x, y} is valid.
        
        # Optimization: Only try if best_size + 1 > best_size (always true)
        # We need a fast validity check for the candidate set.
        # We can use is_valid_mask.
        
        # To limit search space, we can try removing u and adding the two smallest 
        # available x, y that don't conflict?
        # Or just iterate.
        
        # Given N=256, |S|~50, |Pool|~200.
        # Pairs (x,y) ~ 20000.
        # u ~ 50.
        # Total checks ~ 1M.
        # Each check is O(N^2) ~ 64k ops? No, is_valid_mask is O(N^2) roughly.
        # 1M * 64k is too slow.
        
        # We need a faster way or fewer checks.
        # Let's try: For each u, remove it. 
        # Then try to add two elements greedily to the reduced set.
        
        for u in cur_list:
            if not time_left():
                break
            trial_mask = best_mask & ~(1 << u)
            
            # Try to add two elements to trial_mask
            # Greedy: pick x, then pick y.
            # Or iterate x, then y.
            
            # Let's just try to add ONE element first to see if we can recover size?
            # No, we want to increase size.
            
            # Let's try adding x, then y.
            # We can use try_add on trial_mask.
            
            # Find best x, y to add to trial_mask.
            # This is a 2-step greedy.
            
            found_pair = False
            # We can limit the search to the first few candidates that pass try_add
            # on trial_mask.
            
            candidates = []
            for x in pool:
                if (trial_mask >> x) & 1:
                    continue
                nm_x, ok_x = try_add(trial_mask, x)
                if ok_x:
                    candidates.append((x, nm_x))
                    if len(candidates) > 10: # Limit
                        break
            
            for x, mask_x in candidates:
                # Now try to add y to mask_x
                for y in pool:
                    if y == x:
                        continue
                    if (mask_x >> y) & 1:
                        continue
                    nm_xy, ok_xy = try_add(mask_x, y)
                    if ok_xy:
                        # We have a valid set: (S - {u}) + {x, y}
                        # Size is best_size - 1 + 2 = best_size + 1
                        if nm_xy.bit_count() > best_size:
                            best_mask = nm_xy
                            best_size = nm_xy.bit_count()
                            improved = True
                            found_pair = True
                            cur_list = sorted(from_mask(best_mask))
                            pool = sorted(set(range(1, n + 1)) - set(cur_list))
                            break
                if found_pair:
                    break
            
            if found_pair:
                break
                
    # ---- Phase 3: Swap search -- replace one AP element with a smaller one. ----
    if time_left():
        cur = best_mask
        cur_list = sorted(from_mask(cur))
        improved = True
        while improved and time_left():
            improved = False
            for a in list(cur_list):
                if not time.time() < deadline:
                    break
                for b in range(1, a):
                    if (cur >> b) & 1:
                        continue
                    trial = try_remove(cur, a)
                    nm, ok = try_add(trial, b)
                    if ok:
                        if nm.bit_count() > best_size:
                            best_mask = nm
                            best_size = nm.bit_count()
                            improved = True
                        break
>>>>>>> REPLACE

I also need to ensure the initial seed is good. The AP is okay, but let's try to generate a Greedy Descending seed as well and pick the better one.

Edit 3: Modify seed generation.

