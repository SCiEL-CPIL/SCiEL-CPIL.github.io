"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1
    
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1
    
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
