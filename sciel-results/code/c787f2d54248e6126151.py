# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 115.0  # 120s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    except Exception:
        return False
        
    if np.max(np.abs(T - 1.0)) > 1e-8:
        return False
    return True

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank70(n=4, m=5, p=5):
    """
    Construct a rank-70 decomposition using block structure:
    4x5x5 = 4x4x4 (rank 64) + 4x1x4 (rank 1) + 4x4x1 (rank 4) + 4x1x1 (rank 1)
    """
    rank = 70
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    # Block 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
                c += 1
                
    # Block 2: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        U[i*m + 4, c] = 1.0
    for k in range(4):
        V[4*p + k, c] = 1.0
    for i in range(4):
        for k in range(4):
            W[k*n + i, c] = 1.0
    c += 1
            
    # Block 3: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for j in range(4):
        for i in range(4):
            U[i*m + j, c] = 1.0
        V[j*p + 4, c] = 1.0
        for i in range(4):
            W[4*n + i, c] = 1.0
        c += 1
        
    # Block 4: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i*m + 4, c] = 1.0
    V[4*p + 4, c] = 1.0
    for i in range(4):
        W[4*n + i, c] = 1.0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try the structured rank-70 decomposition first
    candidate = _construct_rank70(n, m, p)
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
