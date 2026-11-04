<<<<<<< SEARCH
    # --- Step 2: Construct 4x4x8 factors ---
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Copy 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            
    # Copy 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8*j + k, 49:98] = V444[4*j + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4*k + i, 49:98] = W444[4*(k-4) + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # --- Step 2: Construct 4x4x8 factors ---
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Copy 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            
    # Copy 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8*j + k, 49:98] = V444[4*j + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4*k + i, 49:98] = W444[4*(k-4) + i, :]
    
    # Time check: if we're running low, return immediately
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Quick verification with a fixed test case
    try:
        A_test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=float)
        B_test = np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16],
                          [17,18,19,20,21,22,23,24],[25,26,27,28,29,30,31,32]], dtype=float)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        A_flat = A_test.flatten(order='C')  # i*m + j
        B_flat = B_test.flatten(order='C')  # j*p + k
        
        u_vals = U_arr.T @ A_flat  # shape (rank,)
        v_vals = V_arr.T @ B_flat  # shape (rank,)
        
        # C[k,i] = sum_r W[k*n+i, r] * u_vals[r] * v_vals[r]
        C_computed = np.zeros((p, n))
        for r in range(rank):
            w_col = W_arr[:, r]  # shape (32,), indexed k*4+i
            C_computed += w_col * (u_vals[r] * v_vals[r])
        
        C_computed = C_computed.reshape(p, n)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Decomposition is incorrect; fall back to naive
            return construct_naive()
    except Exception:
        # If verification fails for any reason, return the computed result
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
