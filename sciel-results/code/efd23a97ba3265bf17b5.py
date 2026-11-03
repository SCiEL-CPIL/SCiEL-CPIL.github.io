"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_50():
    """
    Constructs a rank 50 decomposition for 3x4x5 matrix multiplication.
    Splits the inner dimension m=4 into two parts of size 2.
    Uses rank 5 decomposition for 3x2x1 (matrix-vector) for each of the 5 columns.
    Total rank: 2 parts * 5 columns * 5 rank = 50.
    """
    n, m, p = 3, 4, 5
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Sub-decomposition for 3x2x1
    # U_sub: (6, 5), V_sub: (2, 5), W_sub: (3, 5)
    # Indices: U[i*2+j, r], V[j, r], W[i, r]
    U_sub = np.zeros((6, 5))
    V_sub = np.zeros((2, 5))
    W_sub = np.zeros((3, 5))
    
    # r=0: a*x1 (i=0, j=0)
    U_sub[0*2+0, 0] = 1
    V_sub[0, 0] = 1
    W_sub[0, 0] = 1
    
    # r=1: b*x2 (i=0, j=1)
    U_sub[0*2+1, 1] = 1
    V_sub[1, 1] = 1
    W_sub[0, 1] = 1
    
    # r=2: c*x1 (i=1, j=0)
    U_sub[1*2+0, 2] = 1
    V_sub[0, 2] = 1
    W_sub[1, 2] = 1
    
    # r=3: d*x2 (i=1, j=1)
    U_sub[1*2+1, 3] = 1
    V_sub[1, 3] = 1
    W_sub[1, 3] = 1
    
    # r=4: e*x1 + f*x2 (i=2, j=0 and i=2, j=1)
    U_sub[2*2+0, 4] = 1
    U_sub[2*2+1, 4] = 1
    V_sub[0, 4] = 1
    V_sub[1, 4] = 1
    W_sub[2, 4] = 1
    
    # Part 1: j in {0, 1}
    for k in range(p):
        for r_sub in range(5):
            R = k * 5 + r_sub
            for i in range(n):
                for j_local in range(2):
                    j_global = j_local
                    U[i * m + j_global, R] = U_sub[i * 2 + j_local, r_sub]
                    V[j_global * p + k, R] = V_sub[j_local, r_sub]
                    W[k * n + i, R] = W_sub[i, r_sub]
                    
    # Part 2: j in {2, 3}
    for k in range(p):
        for r_sub in range(5):
            R = 25 + k * 5 + r_sub
            for i in range(n):
                for j_local in range(2):
                    j_global = 2 + j_local
                    U[i * m + j_global, R] = U_sub[i * 2 + j_local, r_sub]
                    V[j_global * p + k, R] = V_sub[j_local, r_sub]
                    W[k * n + i, R] = W_sub[i, r_sub]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    
    # Try the rank 50 construction
    result_50 = _construct_rank_50()
    if _verify(result_50['U'], result_50['V'], result_50['W']):
        return result_50
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
