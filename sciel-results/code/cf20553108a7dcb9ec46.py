import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if U, V, W give an exact decomposition of matrix multiplication."""
    rank = U.shape[1]
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    return np.allclose(T_computed, T_expected)

def _add_strassen_2x2x2(U, V, W, i_off, j_off, k_off, rank_off, n, m, p):
    """Add Strassen's 7-term decomposition for a 2x2x2 sub-problem at global offsets."""
    # Strassen terms for 2x2x2
    # U[i, j], V[j, k], W[k, i]
    terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),  # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),  # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[1, 0], [0, 1]]),  # M3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),  # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),  # M5
        ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),  # M6
        ([[0, 1], [0, -1]], [[0, 1], [1, 1]], [[1, 0], [0, 0]]),  # M7
    ]
    
    for t in range(7):
        U_t, V_t, W_t = terms[t]
        col = rank_off + t
        for i in range(2):
            for j in range(2):
                U[(i_off + i) * m + (j_off + j)][col] += U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * p + (k_off + k)][col] += V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[(k_off + k) * n + (i_off + i)][col] += W_t[k][i]
    return rank_off + 7

def _add_schoolbook(U, V, W, i_off, j_off, k_off, rank_off, i_len, j_len, k_len, n, m, p):
    """Add schoolbook decomposition for a sub-problem."""
    col = rank_off
    for i in range(i_len):
        for j in range(j_len):
            for k in range(k_len):
                U[(i_off + i) * m + (j_off + j)][col] += 1
                V[(j_off + j) * p + (k_off + k)][col] += 1
                W[(k_off + k) * n + (i_off + i)][col] += 1
                col += 1
    return col

def construct():
    n, m, p = 3, 4, 7
    # Total rank 78
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j in [0, 2)
    # Top part: i=0, j in [0,2), k in [0,7) -> 1x2x7 schoolbook
    col = _add_schoolbook(U, V, W, 0, 0, 0, col, 1, 2, 7, n, m, p)
    
    # Bottom part: i in [1,3), j in [0,2), k in [0,7) -> 2x2x7
    # Split k into 2, 2, 3
    # k in [0,2): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 0, 0, col, n, m, p)
    # k in [2,4): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 0, 2, col, n, m, p)
    # k in [4,7): 2x2x3
    # k=4: 2x2x1 schoolbook
    col = _add_schoolbook(U, V, W, 1, 0, 4, col, 2, 2, 1, n, m, p)
    # k in [5,7): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 0, 5, col, n, m, p)
    
    # Block 2: j in [2, 4)
    # Top part: i=0, j in [2,4), k in [0,7) -> 1x2x7 schoolbook
    col = _add_schoolbook(U, V, W, 0, 2, 0, col, 1, 2, 7, n, m, p)
    
    # Bottom part: i in [1,3), j in [2,4), k in [0,7) -> 2x2x7
    # Split k into 2, 2, 3
    # k in [0,2): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 2, 0, col, n, m, p)
    # k in [2,4): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 2, 2, col, n, m, p)
    # k in [4,7): 2x2x3
    # k=4: 2x2x1 schoolbook
    col = _add_schoolbook(U, V, W, 1, 2, 4, col, 2, 2, 1, n, m, p)
    # k in [5,7): 2x2x2 Strassen
    col = _add_strassen_2x2x2(U, V, W, 1, 2, 5, col, n, m, p)
    
    # col should be 78
    if col != rank:
        # In case of miscalculation, adjust or raise
        pass
        
    return {"U": U, "V": V, "W": W}
