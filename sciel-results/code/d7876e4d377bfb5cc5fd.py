import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_dec[i, k] = val
        
        if not np.allclose(C, C_dec, atol=tol):
            return False
    return True

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2 by 2x2 matrix multiplication.
    Convention:
      U indexed by (i, j) with row index i*m + j, m=2
      V indexed by (j, k) with row index j*p + k, p=2
      W indexed by (k, i) with row index k*n + i, n=2
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10 - a00)(b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01 - a11)(b11 + b10)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 120
    
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    # Split 11 columns into 5 blocks of 2 and 1 block of 1
    total_rank = 5 * 7 + 1 * 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # 5 Strassen blocks for columns 0-9
    for block_idx in range(5):
        start_col = block_idx * 2
        U[:, col:col+7] = U_base
        for j in range(2):
            for k_local in range(2):
                global_k = start_col + k_local
                V[j * p + global_k, col:col+7] = V_base[j * 2 + k_local, :]
        for k_local in range(2):
            for i in range(2):
                global_k = start_col + k_local
                W[global_k * n + i, col:col+7] = W_base[k_local * 2 + i, :]
        col += 7
    
    # 1 schoolbook block for column 10
    start_col = 10
    U_sub = np.zeros((4, 4))
    V_sub = np.zeros((2, 4))
    W_sub = np.zeros((2, 4))
    
    # (i=0, j=0): a00 * b_{0,10}
    U_sub[0, 0] = 1
    V_sub[0, 0] = 1
    W_sub[0, 0] = 1
    
    # (i=0, j=1): a01 * b_{1,10}
    U_sub[1, 1] = 1
    V_sub[1, 1] = 1
    W_sub[0, 1] = 1
    
    # (i=1, j=0): a10 * b_{0,10}
    U_sub[2, 2] = 1
    V_sub[0, 2] = 1
    W_sub[1, 2] = 1
    
    # (i=1, j=1): a11 * b_{1,10}
    U_sub[3, 3] = 1
    V_sub[1, 3] = 1
    W_sub[1, 3] = 1
    
    U[:, col:col+4] = U_sub
    for j in range(2):
        V[j * p + start_col, col:col+4] = V_sub[j, :]
    for i in range(2):
        W[start_col * n + i, col:col+4] = W_sub[i, :]
    col += 4
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n*m, rank))
            V_fb = np.zeros((m*p, rank))
            W_fb = np.zeros((p*n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, column] = 1
                        V_fb[j*p+k, column] = 1
                        W_fb[k*n+i, column] = 1
                        column += 1
            U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
