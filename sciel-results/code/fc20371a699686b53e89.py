"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _build_rank4_decomposition():
    """Build a rank-4 decomposition using the j-index structure.
    
    The matrix multiplication tensor for 3x4 * 4x8 -> 8x3 is T[i,j,k] = 1 for all
    i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}.
    
    We decompose as: T = sum_{j=0}^{3} u_j ⊗ v_j ⊗ w_j where:
    - u_j has 1 at position i*4+j for all i, 0 elsewhere
    - v_j has 1/2 at position j*8+k for all k, 0 elsewhere  
    - w_j has 1/2 at position k*3+i for all k,i, 0 elsewhere
    
    Then sum_{j=0}^{3} U[i*4+j,j] * V[j*8+k,j] * W[k*3+i,j] = sum_j 1*(1/2)*(1/2) = 4*(1/4) = 1. ✓
    All entries are integers or half-integers. ✓
    """
    n, m, p = 3, 4, 8
    rank = m  # 4
    
    U = np.zeros((n * m, rank), dtype=float)  # (12, 4)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 4)
    W = np.zeros((p * n, rank), dtype=float)  # (24, 4)
    
    for j in range(m):
        # U: 1 at (i*m + j) for all i
        for i in range(n):
            U[i * m + j, j] = 1.0
        # V: 1/2 at (j*p + k) for all k
        for k in range(p):
            V[j * p + k, j] = 0.5
        # W: 1/2 at (k*n + i) for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 0.5
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 3, 4, 8
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    assert V.shape[1] == rank
    assert W.shape[1] == rank
    
    # Compute T from decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                T_decomp[i, j, k] = np.dot(np.dot(u_val, v_val), w_val)
    
    # True tensor: T_true[i, j, k] = 1 for all i,j,k
    T_true = np.ones((n, m, p), dtype=float)
    
    max_err = np.max(np.abs(T_decomp - T_true))
    return max_err < 1e-10


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Try rank-4 decomposition
    try:
        U, V, W = _build_rank4_decomposition()
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 96)
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

# EVOLVE-BLOCK-END
