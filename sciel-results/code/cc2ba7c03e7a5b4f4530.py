<<<<<<< SEARCH
def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try chaotic sequences with various parameters
    chaotic_params = [
        (3.99, 0.12345),
        (3.999, 0.12345),
        (3.99, 0.45678),
        (3.999, 0.45678),
        (3.99, 0.78901),
        (3.999, 0.78901),
        (3.95, 0.12345),
        (3.95, 0.45678),
    ]
    for r, x0 in chaotic_params:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        h_chaos = _chaotic_sequence(N, r=r, x0=x0)
        bound = _compute_c3_bound(h_chaos)
        if bound < best_bound:
            best_bound = bound
            best_h = h_chaos.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(15):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 50)
        best_h, best_bound = _local_search(best_h, max_iter=5000, time_budget=local_time)
    
    # Pairwise flip search to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        pair_time = min(remaining - 15, 30)
        best_h, best_bound = _pairwise_flip_search(best_h, max_iter=2000, time_budget=pair_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Mix of random and chaotic initializations
            if trial % 2 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            else:
                h = _chaotic_sequence(N, r=3.99 + 0.001 * trial, x0=0.1 + 0.01 * trial)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 15)
                h, bound = _local_search(h, max_iter=2000, time_budget=local_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final local polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        polish_time = min(remaining - 5, 15)
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_budget=polish_time)
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h
=======
def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try chaotic sequences with various parameters
    chaotic_params = [
        (3.99, 0.12345),
        (3.999, 0.12345),
        (3.99, 0.45678),
        (3.999, 0.45678),
        (3.99, 0.78901),
        (3.999, 0.78901),
        (3.95, 0.12345),
        (3.95, 0.45678),
        (3.99, 0.23456),
        (3.999, 0.34567),
    ]
    for r, x0 in chaotic_params:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        h_chaos = _chaotic_sequence(N, r=r, x0=x0)
        bound = _compute_c3_bound(h_chaos)
        if bound < best_bound:
            best_bound = bound
            best_h = h_chaos.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 50)
        best_h, best_bound = _local_search(best_h, max_iter=5000, time_budget=local_time)
    
    # Pairwise flip search to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        pair_time = min(remaining - 15, 30)
        best_h, best_bound = _pairwise_flip_search(best_h, max_iter=2000, time_budget=pair_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Mix of random and chaotic initializations
            if trial % 2 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            else:
                h = _chaotic_sequence(N, r=3.99 + 0.001 * trial, x0=0.1 + 0.01 * trial)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 15)
                h, bound = _local_search(h, max_iter=2000, time_budget=local_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final local polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        polish_time = min(remaining - 5, 15)
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_budget=polish_time)
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h
>>>>>>> REPLACE
