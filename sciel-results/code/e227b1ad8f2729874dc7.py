<<<<<<< SEARCH
    # Simulated Annealing
    if time.time() < deadline - 5.0:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, max_iters=3000)
        sa_ms = _makespan(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts if time permits
    rng = np.random.default_rng(42)
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() >= deadline - 2.0:
                break
            random_perm = rng.permutation(n)
            ms_r = _makespan(random_perm, pt)
            if ms_r < best_ms:
                best_perm = random_perm
                best_ms = ms_r
            # Local search from random start
            ls_r = _local_search(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
    
    return list(best_perm)
=======
    # Tabu search
    if time.time() < deadline - 5.0:
        ts_perm = _tabu_search(best_perm, pt, deadline, max_iters=5000)
        ts_ms = _makespan(ts_perm, pt)
        if ts_ms < best_ms:
            best_perm = ts_perm
            best_ms = ts_ms
    
    # Random restarts if time permits
    rng = np.random.default_rng(42)
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() >= deadline - 2.0:
                break
            random_perm = rng.permutation(n)
            ms_r = _makespan(random_perm, pt)
            if ms_r < best_ms:
                best_perm = random_perm
                best_ms = ms_r
            # Local search from random start
            ls_r = _local_search(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
    
    return list(best_perm)
>>>>>>> REPLACE
