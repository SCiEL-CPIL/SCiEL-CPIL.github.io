"""Matrix multiplication 2x2x8: rank-minimal exact tensor decomposition.

Uses a Strassen-style construction to achieve rank 28 (vs schoolbook 32).
The 2x2x8 tensor is decomposed into 4 independent 2x2x2 multiplications
(blocks of 2 columns), each handled by Strassen's 7-term algorithm.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    
    Convention:
      U shape (4, 7): U[i*2+j, t] = coeff of A[i,j] in term t
      V shape (4, 7): V[j*2+k, t] = coeff of B[j,k] in term t  
      W shape (4, 7): W[k*2+i, t] = coeff of C[i,k] in term t
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[0]=C00(k0i0), W[3]=C11(k1i1)
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # W[1]=C10(k0i1), W[3]=C11(k1i1)
    
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # W[2]=C01(k1i0), W[3]=C11(k1i1)
    
    # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # W[0]=C00(k0i0), W[1]=C10(k0i1)
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # W[0]=C00, W[2]=C01
    
    # M6 = (A10-A00)(B00+B10) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # W[3]=C11
    
    # M7 = (A01-A11)(B01+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # W[0]=C00
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    U_b, V_b, W_b = _strassen_2x2x2()
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for block in range(4):
        off = block * 7
        # U is shared across blocks
        U[:, off:off+7] = U_b
        
        # V: global column index = 2*block + local_col
        for j in range(m):
            for kc in range(2):
                kg = 2 * block + kc
                V[j * p + kg, off:off+7] = V_b[j * 2 + kc, :]
        
        # W: global column index for C = 2*block + local_col
        for kc in range(2):
            kg = 2 * block + kc
            for i in range(n):
                W[kg * n + i, off:off+7] = W_b[kc * 2 + i, :]
    
    if _verify(U, V, W, n, m, p):
        U_list = [[int(U[i, t]) for t in range(rank)] for i in range(n*m)]
        V_list = [[int(V[i, t]) for t in range(rank)] for i in range(m*p)]
        W_list = [[int(W[i, t]) for t in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 32
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
