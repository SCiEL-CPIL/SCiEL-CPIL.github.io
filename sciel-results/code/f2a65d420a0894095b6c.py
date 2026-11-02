"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm on pairs of columns. For p=12, we have 6 pairs,
    each requiring rank 7, giving total rank = 42.
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00 + A11) * (B00 + B11)
                U[0][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 1:
                # M2 = (A10 + A11) * B00
                U[2][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                W[2*t*n + 1][c] = 1
                W[(2*t+1)*n + 1][c] = -1
            elif s == 2:
                # M3 = A00 * (B01 - B11)
                U[0][c] = 1
                V[0*p + 2*t+1][c] = 1
                V[1*p + 2*t+1][c] = -1
                W[(2*t+1)*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 3:
                # M4 = A11 * (B10 - B00)
                U[3][c] = 1
                V[1*p + 2*t][c] = 1
                V[0*p + 2*t][c] = -1
                W[2*t*n + 0][c] = 1
                W[2*t*n + 1][c] = 1
            elif s == 4:
                # M5 = (A00 + A01) * B11
                U[0][c] = 1
                U[1][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = -1
                W[(2*t+1)*n + 0][c] = 1
            elif s == 5:
                # M6 = (A10 - A00) * (B00 + B01)
                U[2][c] = 1
                U[0][c] = -1
                V[0*p + 2*t][c] = 1
                V[0*p + 2*t+1][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 6:
                # M7 = (A01 - A11) * (B10 + B11)
                U[1][c] = 1
                U[3][c] = -1
                V[1*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
