<<<<<<< SEARCH
def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    T = _target_tensor()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = len(U_sb[0])
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try to reduce rank from 44 down to 30
    for r in range(44, 29, -1):
        if _time_remaining() < 2.0:
            break
            
        # Strategy: Try multiple random subsets of the schoolbook structure
        # Or just run ALS with different seeds
        found = False
        for attempt in range(5):
            if _time_remaining() < 1.0:
                break
                
            # Initialize factors using schoolbook columns
            # We take the first r columns of schoolbook as a starting point
            # But since schoolbook columns are orthogonal, taking a subset might not be a good start for a general rank-r tensor.
            # Better: Random initialization
            
            U, V, W = _solve_for_factors(T, range(r), r)
            
            if U is None:
                continue
                
            # Snap to half-integers
            U_r = _round_to_half_integer(U)
            V_r = _round_to_half_integer(V)
            W_r = _round_to_half_integer(W)
            
            if U_r is not None and V_r is not None and W_r is not None:
                if _verify_exact(U_r, V_r, W_r):
                    best_rank = r
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    found = True
                    break # Found a valid decomposition for this rank, try lower?
                    # Actually, we want the MINIMAL rank. So we should break the inner loop and continue to r-1.
                    
        # If we found a valid rank r, we can break the outer loop? 
        # No, we want the lowest rank. So we continue to r-1.
        # But if we found rank 44, we should check 43.
        
    return best_result
=======
def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try structured decomposition
    structured = _build_structured_decomposition()
    if structured is not None:
        # Verify it's valid and has lower rank
        U_s = np.array(structured["U"])
        V_s = np.array(structured["V"])
        W_s = np.array(structured["W"])
        rank_s = U_s.shape[1]
        rank_sb = U_sb.shape[1]
        
        if rank_s < rank_sb and _verify_exact(U_s, V_s, W_s):
            best_result = structured
    
    return best_result
>>>>>>> REPLACE

