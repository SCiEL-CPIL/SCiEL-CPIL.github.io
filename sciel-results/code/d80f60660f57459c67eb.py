<<<<<<< SEARCH
    # Try to find a better rank than 44
    best_result = None
    
    for R in [40, 41, 42, 43]:
        if time.time() - start_time > timeout:
            break
        result = try_rank(R, max_restarts=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        best_result = get_fallback()
        
    return best_result
=======
    # First, verify and return the known rank-44 solution
    fallback = get_fallback()
    
    # Verify the fallback is correct
    U_fb = np.array(fallback["U"]).T  # (R, size_U)
    V_fb = np.array(fallback["V"]).T  # (R, size_V)
    W_fb = np.array(fallback["W"]).T  # (R, size_W)
    res_fb = compute_residuals(U_fb, V_fb, W_fb)
    if np.max(np.abs(res_fb)) > 1e-9:
        # Fallback is broken; construct a simple schoolbook rank-48
        rank = 48
        U = np.zeros((size_U, rank))
        V = np.zeros((size_V, rank))
        W = np.zeros((size_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1
        # Fill remaining with zeros (shouldn't happen for 2x3x8 with j==k: 2*3*8=48 terms)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = fallback
    best_rank = 44
    
    # Try to find a better rank using the remaining time
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        for R in [43, 42, 41, 40]:
            if time.time() - start_time > timeout - 5:
                break
            result = try_rank(R, max_restarts=5)
            if result is not None:
                best_result = result
                best_rank = R
                break
    
    return best_result
>>>>>>> REPLACE

Now let me also improve the `try_rank` function to use a better optimization strategy and the `get_fallback` to be more robust:

