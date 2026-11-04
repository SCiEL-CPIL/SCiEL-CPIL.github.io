<<<<<<< SEARCH
def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try to find rank < 48
    # Start from 47 down to 30
    for R in range(47, 29, -1):
        if time.time() - start_time > deadline:
            break
        result = _try_rank(R, n, m, p, timeout=5.0)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}
=======
def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Build the structured decomposition (rank 44)
    U, V, W = _build_decomposition()
    
    # Verify
    err = _verify_fast(U, V, W, n, m, p)
    if err < 1e-9:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
