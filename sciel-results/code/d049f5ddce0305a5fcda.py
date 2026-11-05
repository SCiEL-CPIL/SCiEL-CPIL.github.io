<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # 2 blocks in m dimension (size 2 each)
    # 4 blocks in p dimension (size 2 each)
    # Total 8 blocks, each rank 7. Total rank 56.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = _build_strassen_block(j0, k0, col, U, V, W, n, m, p)
            
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }
=======
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # 2 blocks in m dimension (size 2 each)
    # 4 blocks in p dimension (size 2 each)
    # Total 8 blocks, each rank 7. Total rank 56.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = _build_strassen_block(j0, k0, col, U, V, W, n, m, p)
            
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
        return {
            "U": U,
            "V": V,
            "W": W
        }
    
    # Try to reduce rank using lattice reduction / null space approach
    if _time_left() > 30:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p)
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
>>>>>>> REPLACE
