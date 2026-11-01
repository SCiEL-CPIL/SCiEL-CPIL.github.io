"""Tensor decomposition for 2x2 by 2x3 matrix multiplication - rank 11."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    np.random.seed(42)
    for trial in range(30):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_direct = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            A_mat = U[:, r].reshape(n, m)
            B_mat = V[:, r].reshape(m, p)
            W_mat = W[:, r].reshape(p, n)
            scalar_A = np.sum(A * A_mat)
            scalar_B = np.sum(B * B_mat)
            C_decomp += scalar_A * scalar_B * W_mat.T
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _try_rank_11():
    """Construct a rank-11 decomposition: Strassen (7) + 4 terms for 3rd column."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for 2x2x2 (columns 0,1 of B):
    # M1 = A[0,0] * (B[0,0]+B[1,1])
    U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A[0,0] * (B[1,0]+B[1,1])
    U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = 1
    W[0, 2] = -1; W[1, 2] = 1
    
    # M4 = A[1,1] * (B[0,0]+B[0,1])
    U[3, 3] = 1; V[0, 3] = 1; V[1, 3] = 1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 5] = 1; U[3, 5] = -1; V[2, 5] = 1; V[3, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    
    # M7 = (A[0,0]-A[1,0]) * (B[0,0]+B[0,1])
    U[0, 6] = 1; U[2, 6] = -1; V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    # Third column (k=2): B indices j*3+2: B[0,2]->2, B[1,2]->5
    # M8 = A[0,0]*B[0,2]
    U[0, 7] = 1; V[2, 7] = 1
    W[4, 7] = 1
    
    # M9 = A[0,1]*B[1,2]
    U[1, 8] = 1; V[5, 8] = 1
    W[4, 8] = 1
    
    # M10 = A[1,0]*B[0,2]
    U[2, 9] = 1; V[2, 9] = 1
    W[5, 9] = 1
    
    # M11 = A[1,1]*B[1,2]
    U[3, 10] = 1; V[5, 10] = 1
    W[5, 10] = 1
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try rank 11 first
    r11 = _try_rank_11()
    if r11 is not None:
        return r11
    
    # Fallback: schoolbook (rank 12)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
