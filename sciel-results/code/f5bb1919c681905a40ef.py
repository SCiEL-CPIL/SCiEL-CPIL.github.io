def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try progressively lower ranks
    for target_rank in range(20, 27):
        if time.time() - start_time > timeout - 10:
            break
        
        # Try multiple seeds for each rank
        # Allocate remaining time to this rank
        time_left = timeout - 10 - (time.time() - start_time)
        if time_left <= 0:
            break
            
        num_seeds = 3
        time_per_seed = time_left / num_seeds
        
        found = False
        for seed in range(num_seeds):
            if time.time() - start_time > timeout - 10:
                break
            res = _search_rank_numpy(target_rank, n, m, p, time_budget=time_per_seed, seed=seed)
            if res is not None:
                U, V, W = res
                # Verify
                if _verify(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best
