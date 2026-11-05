    while time.time() < deadline:
        # Phase 1: Greedy Fill with Randomization
        S = list(fixed)
        random.shuffle(S) # Fixed elements order doesn't matter for sums, but let's keep them
        # Actually, just start with fixed
        S = list(fixed)
        sums = set()
        for i, x in enumerate(S):
            for y in S[i:]:
                sums.add(x + y)
                
        # Randomize candidate order
        candidates = [x for x in range(1, 285) if x not in fixed]
        random.shuffle(candidates)
        
        for x in candidates:
            new_sums = check_add(S, sums, x)
            if new_sums is not None:
                S.append(x)
                sums |= new_sums
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            if best_size == 20:
                return best_set
        
        # Phase 2: Local Search (Swaps)
        # Only if we didn't reach 20
        while time.time() < deadline:
            candidates = [x for x in range(1, 285) if x not in S]
            if not candidates:
                break
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                new_sums = check_add(S, sums, x)
                if new_sums is not None:
                    S.append(x)
                    sums |= new_sums
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                        if best_size == 20:
                            return best_set
                    break # Restart local search loop with new state
                    
            if not added:
                # Try swap
                removable = [x for x in S if x not in fixed]
                if not removable:
                    break # Stuck, restart Phase 1
                    
                y = random.choice(removable)
                S_backup = S[:]
                sums_backup = sums.copy()
                
                # Remove y
                S.remove(y)
                for z in S:
                    sums.discard(y + z)
                sums.discard(2 * y)
                
                # Try to add a new element
                candidates = [x for x in range(1, 285) if x not in S]
                random.shuffle(candidates)
                
                swap_success = False
                for x in candidates:
                    new_sums = check_add(S, sums, x)
                    if new_sums is not None:
                        S.append(x)
                        sums |= new_sums
                        swap_success = True
                        break
                        
                if not swap_success:
                    # Rollback
                    S = S_backup
                    sums = sums_backup
                    # If we can't swap, we are likely stuck. Break to restart Phase 1.
                    break
