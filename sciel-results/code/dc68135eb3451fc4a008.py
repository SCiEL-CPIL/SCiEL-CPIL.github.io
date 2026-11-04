<<<<<<< SEARCH
    # First try the Strassen block approach (rank 33)
    strassen_result = try_strassen_blocks()
    if strassen_result is not None:
        U, V, W, rank = strassen_result
        return {"U": U, "V": V, "W": W}
    
    # Try the Kronecker structure approach
    kronecker_result = try_kronecker_structure(24)
    if kronecker_result is not None:
        U, V, W, rank = kronecker_result
        return {"U": U, "V": V, "W": W}
    
    # Try direct ALS for various ranks
    for rank in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        res = try_direct_als(rank, max_iters=150, trials=2)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
=======
    # First try the Strassen block approach (rank 33)
    strassen_result = try_strassen_blocks()
    if strassen_result is not None:
        U, V, W, rank = strassen_result
        return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}
    
    # Try the Kronecker structure approach
    kronecker_result = try_kronecker_structure(24)
    if kronecker_result is not None:
        U, V, W, rank = kronecker_result
        return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}
    
    # Try direct ALS for various ranks
    for rank in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        res = try_direct_als(rank, max_iters=150, trials=2)
        if res:
            U, V, W = res
            return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}
            
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": np.array(U_fb), "V": np.array(V_fb), "W": np.array(W_fb)}
>>>>>>> REPLACE
