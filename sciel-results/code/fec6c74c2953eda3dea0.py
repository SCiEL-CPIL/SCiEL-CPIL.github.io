def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a trivial decomposition with rank 30.
    Each term corresponds to a specific intermediate index j (0..2) and output column k (0..9).
    """
    n, m, p = 2, 3, 10
    
    rank = m * p  # 3 * 10 = 30
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U vector: 1 at indices (i, j) for i=0..n-1
            # Index in U is i * m + j
            for i in range(n):
                U[i * m + j, r] = 1
            
            # V vector: 1 at index (j, k)
            # Index in V is j * p + k
            V[j * p + k, r] = 1
            
            # W vector: 1 at indices (k, i') for i'=0..n-1
            # Index in W is k * n + i'
            for i_prime in range(n):
                W[k * n + i_prime, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
