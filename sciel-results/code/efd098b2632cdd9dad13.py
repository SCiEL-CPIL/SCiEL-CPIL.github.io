def _rank25_decomposition(n, m, p):
    """
    Construct a rank-25 decomposition for 2x5x5 matrix multiplication.
    """
    if n != 2 or m != 5 or p != 5:
        return None
    
    rank = m * p # 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 25 construction
    res25 = _rank25_decomposition(n, m, p)
    if res25 is not None:
        if _verify(res25["U"], res25["V"], res25["W"], n, m, p):
            return res25
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
