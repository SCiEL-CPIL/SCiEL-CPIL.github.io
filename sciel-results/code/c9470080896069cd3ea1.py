<<<<<<< SEARCH
    # --- Phase 2: Error-guided local search ---
    current = best_network[:]
    # Pre-compute all valid (i,j) pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for iteration in range(20000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, find the violation and try to repair
        violation = find_violation(candidate)
        repaired = False
        if violation:
            vi, vj = violation
            # Try inserting comparators involving the violated wires
            repair_candidates = []
            for w in range(N):
                if w != vi and w != vj:
                    # (w, vi) if w < vi, else (vi, w)
                    if w < vi:
                        repair_candidates.append((w, vi))
                    else:
                        repair_candidates.append((vi, w))
                if w != vj:
                    if w < vj:
                        repair_candidates.append((w, vj))
                    else:
                        repair_candidates.append((vj, w))
            # Also try (vi, vj) itself
            repair_candidates.append((vi, vj))
            
            random.shuffle(repair_candidates)
            
            for (i, j) in repair_candidates[:30]:
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                # Try inserting at a random position
                pos = random.randint(0, len(candidate))
                temp = candidate[:]
                temp.insert(pos, (i, j))
                
                if verify_network(temp):
                    current = temp
                    repaired = True
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break
        
        if not repaired:
            # If we can't repair, keep current
            pass
=======
    # --- Phase 2: Error-guided local search with better repair ---
    current = best_network[:]
    # Pre-compute all valid (i,j) pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for iteration in range(50000):
        if (time.time() - start_time) > DEADLINE - 8:
            break
        if len(current) <= 70:
            break
            
        # Random removal - try removing from random positions
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, find the violation and try to repair
        violation = find_violation(candidate)
        repaired = False
        if violation:
            vi, vj = violation
            # Try inserting comparators involving the violated wires
            # Focus on wires that are likely to help fix the inversion
            repair_candidates = []
            # First try (vi, vj) itself - most direct fix
            repair_candidates.append((vi, vj))
            # Then try neighbors of vi and vj
            for w in range(N):
                if w != vi and w != vj:
                    if w < vi:
                        repair_candidates.append((w, vi))
                    else:
                        repair_candidates.append((vi, w))
                if w != vj:
                    if w < vj:
                        repair_candidates.append((w, vj))
                    else:
                        repair_candidates.append((vj, w))
            
            random.shuffle(repair_candidates)
            
            for (i, j) in repair_candidates[:20]:
                if (time.time() - start_time) > DEADLINE - 8:
                    break
                # Try inserting at multiple positions
                for pos in [0, len(candidate) // 2, len(candidate) - 1, random.randint(0, len(candidate))]:
                    temp = candidate[:]
                    temp.insert(pos, (i, j))
                    
                    if verify_network(temp):
                        current = temp
                        repaired = True
                        if len(current) < best_size:
                            best_network = current[:]
                            best_size = len(current)
                        break
                if repaired:
                    break
        
        if not repaired:
            pass
>>>>>>> REPLACE

