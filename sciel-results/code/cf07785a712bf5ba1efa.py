import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    """Return the schoolbook decomposition (rank 36) as fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = 1 iff i_A == i_C, j_A == j_B, k_B == k_C.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i_A in range(n):
        for j_A in range(m):
            for k_B in range(p):
                # i index in U
                i = i_A * m + j_A
                # j index in V
                j = j_A * p + k_B
                # k index in W
                # W index is k_C * n + i_C
                # We need k_B == k_C and i_A == i_C
                k_C = k_B
                i_C = i_A
                k = k_C * n + i_C
                T_expected[i, j, k] = 1.0
                
    return np.allclose(T_decomp, T_expected, atol=1e-10)

def _search_decomposition(n, m, p, max_rank, time_limit):
    start = time.time()
    # Try ranks from max_rank down to 1
    for r in range(max_rank, 1, -1):
        if time.time() > time_limit:
            break
        # Try several random initializations
        for attempt in range(2000):
            if time.time() > time_limit:
                break
            
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Random sparse initialization
            # Each column should have a few non-zeros
            for col in range(r):
                # U
                nz_u = np.random.randint(1, n*m+1)
                idx_u = np.random.choice(n*m, nz_u, replace=False)
                U[idx_u, col] = np.random.choice([-1, 1], nz_u)
                
                # V
                nz_v = np.random.randint(1, m*p+1)
                idx_v = np.random.choice(m*p, nz_v, replace=False)
                V[idx_v, col] = np.random.choice([-1, 1], nz_v)
                
                # W
                nz_w = np.random.randint(1, p*n+1)
                idx_w = np.random.choice(p*n, nz_w, replace=False)
                W[idx_w, col] = np.random.choice([-1, 1], nz_w)
                
            if _verify_decomposition(U, V, W):
                # Convert to list of ints
                U_list = U.astype(int).tolist()
                V_list = V.astype(int).tolist()
                W_list = W.astype(int).tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    return None

def construct():
    start = time.time()
    deadline = start + 110
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    result = _search_decomposition(n, m, p, schoolbook_rank - 1, deadline)
    if result is not None:
        return result
        
    return _schoolbook()
