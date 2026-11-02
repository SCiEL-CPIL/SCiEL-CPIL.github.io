def construct():
    n, m, p = DIMENSIONS
    # Rank 1 decomposition for the all-ones tensor
    # U[i*m+j][0] = 1
    # V[j*p+k][0] = 1
    # W[k*n+i][0] = 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {"U": U_s, "V": V_s, "W": W_s}
