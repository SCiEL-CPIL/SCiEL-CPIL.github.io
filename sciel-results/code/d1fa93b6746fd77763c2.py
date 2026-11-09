"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,abc,dec->ijabde', U_reshaped, V_reshaped, W_reshaped)
        
        idx_i = np.arange(n)[:, None, None, None, None, None]
        idx_j = np.arange(m)[None, :, None, None, None, None]
        idx_a = np.arange(m)[None, None, :, None, None, None]
        idx_b = np.arange(p)[None, None, None, :, None, None]
        idx_d = np.arange(p)[None, None, None, None, :, None]
        idx_e = np.arange(n)[None, None, None, None, None, :]
        
        target = (idx_i == idx_e) & (idx_j == idx_a) & (idx_b == idx_d)
        target = target.astype(float)
        
        if np.allclose(T, target, atol=1e-5):
            return True
        return False
    except Exception:
        return False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # No search is performed to ensure reliability and speed.
    # The schoolbook decomposition is the known safe baseline.
    
    return best_result
# EVOLVE-BLOCK-END
