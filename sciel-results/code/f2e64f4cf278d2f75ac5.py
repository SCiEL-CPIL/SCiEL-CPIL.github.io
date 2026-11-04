<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates (from Inspiration)
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search (from Current)
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Dynamic Local Search (from Inspiration)
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Final Polish: Fine-grained local search on the best set
    # This addresses the "refinement machinery" requirement.
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 10, max_val + 11):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    # Add base-expansion candidates
    for B in [2, 3, 4, 5]:
        for k in [3, 4, 5, 6]:
            if 2**k > 60:
                break
            cand = base_expansion_set(B, k)
            if 0 in cand and len(cand) >= 2:
                structured_candidates.append(cand)
            cand2 = base_expansion_mixed(B, k, sparsity=0.4)
            if 0 in cand2 and len(cand2) >= 2:
                structured_candidates.append(cand2)
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 50:
            break
        for m in range(2, 25):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Dynamic Local Search
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:5]:
        if cand != best_set and 0 in cand:
            seeds.append(list(cand))
    
    max_val_limit = 2000
    max_n_limit = 50
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 3:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 2 and iteration < 1000:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 35 and time.time() < deadline - 2:
                max_val = max(current)
                for new_val in range(max_val - 15, max_val + 16):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 2:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    # Clear cache to save memory
    _score_cache.clear()
    
    return best_set
>>>>>>> REPLACE

