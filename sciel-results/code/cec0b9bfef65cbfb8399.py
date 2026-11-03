<<<<<<< SEARCH
    # Try to reduce the Strassen rank via ALS refinement
    if time.time() < deadline - 60:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if len(U_red[0]) < best_rank:
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
            continue
    
    return best_result
=======
    # Try to reduce the Strassen rank via SVD subspace elimination + ALS refinement
    if time.time() < deadline - 40:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if len(U_red[0]) < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = len(U_red[0])
    
    # Try to find lower-rank decompositions using ALS with multiple seeds
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
        
        if found:
            continue
    
    return best_result
>>>>>>> REPLACE

