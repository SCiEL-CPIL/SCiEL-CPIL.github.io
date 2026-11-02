<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        # Fallback to schoolbook if Strassen construction has issues
        return _schoolbook()
    
    # First, try to reduce the Strassen rank by identifying dependent terms
    if time.time() < deadline - 60:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try to find lower-rank decompositions using ALS
    # Search from best_rank-1 down to 40
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400, deadline=deadline - 20)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, continue searching even lower
            continue
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        # Fallback to schoolbook if Strassen construction has issues
        return _schoolbook()
    
    # First, try to reduce the Strassen rank by identifying dependent terms
    if time.time() < deadline - 40:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try to find lower-rank decompositions using ALS with random starts
    # Search from best_rank-1 down to 40
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=500, deadline=deadline - 15)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if not found:
            # If we can't find rank r, stop (lower ranks are harder)
            break
    
    return best_result
>>>>>>> REPLACE
