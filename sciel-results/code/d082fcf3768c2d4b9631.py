<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

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
        [0, 1, 4, 9, 11], # Known good small set
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    # This is a powerful strategy for small sets where local search might get stuck in local minima.
    
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 50:
        # n=3
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        # n=4
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        # n=5
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Geometric Search
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = [0]
            seen = {0}
            for k in range(1, m + 1):
                val = floor(b ** k)
                if val <= 0:
                    val = 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
            U.sort()
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                best_set = U
                best_score = s

    # 4. Local Search from Best Seeds
    search_seeds = [best_set]
    
    # Add top structured candidates as additional seeds
    for cand in structured_candidates:
        if cand != best_set:
            search_seeds.append(cand)
            
    # Add a few random perturbations of the best set
    for _ in range(5):
        pert = best_set[:]
        if len(pert) > 1:
            idx = random.randint(1, len(pert)-1)
            pert[idx] += random.randint(-5, 5)
            if pert[idx] > 0:
                pert = sorted(set(pert))
                if len(pert) >= 2 and 0 in pert:
                    search_seeds.append(pert)

    for seed in search_seeds:
        if time.time() > deadline - 10:
            break
        local_score, local_set = local_search_dynamic(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    return best_set
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates - expanded with more progression families
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 9, 11], # Known good small set
        # Additional families
        [0, 1, 2, 5, 10, 17, 26, 37], # x^2 + x + 1 style
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular numbers
        [0, 1, 4, 9, 16, 25, 36, 49], # Squares (shorter)
        [0, 1, 5, 11, 19, 29, 41], # 2x^2 + x - 1 style
        [0, 1, 2, 6, 12, 20, 30], # x(x+1) style
        [0, 1, 4, 8, 15, 24, 36], # Mixed progression
        [0, 1, 3, 8, 15, 24, 35], # x^2 - x + 1 style
        [0, 2, 5, 10, 17, 26, 37], # Shifted squares
        [0, 1, 4, 9, 16, 25, 40, 58], # Mixed quadratic
        [0, 1, 3, 7, 14, 24, 37], # x^2 - 3x + 3 style
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    # This is a powerful strategy for small sets where local search might get stuck in local minima.
    
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 50:
        # n=3
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        # n=4
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        # n=5
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Geometric Search
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = [0]
            seen = {0}
            for k in range(1, m + 1):
                val = floor(b ** k)
                if val <= 0:
                    val = 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
            U.sort()
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                best_set = U
                best_score = s

    # 4. Local Search from Best Seeds
    search_seeds = [best_set]
    
    # Add top structured candidates as additional seeds
    for cand in structured_candidates:
        if cand != best_set:
            search_seeds.append(cand)
            
    # Add a few random perturbations of the best set
    for _ in range(8):
        pert = best_set[:]
        if len(pert) > 1:
            # Perturb 1-2 elements
            for _ in range(random.randint(1, 2)):
                idx = random.randint(1, len(pert)-1)
                pert[idx] += random.randint(-8, 8)
                if pert[idx] < 1:
                    pert[idx] = 1
            pert = sorted(set(pert))
            if len(pert) >= 2 and 0 in pert:
                search_seeds.append(pert)

    # Run local search on all seeds with time budget
    for seed in search_seeds:
        if time.time() > deadline - 15:
            break
        local_score, local_set = local_search_dynamic(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # 5. Final Polish Pass - tighter tolerance, more iterations
    if time.time() < deadline - 5:
        polish_score, polish_set = local_search_dynamic(best_set, deadline, polish=True, max_val_limit=5000)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set

    return best_set
>>>>>>> REPLACE

