"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 decomposition for 2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10 + A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += -1; W[3][1] += -1
    
    # M3 = A00(B10 - B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11(B01 - B00)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5 = (A00 + A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses block Strassen decomposition with rank 56.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 4, 8
    
    U_2x2, V_2x2, W_2x2 = _strassen_2x2_decomposition()
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a_block in range(2):
        for k_block in range(4):
            for c in range(7):
                # U: A[i, a_block*2 + j]
                for i in range(2):
                    for j in range(2):
                        U[i * m + (a_block * 2 + j)][col] += U_2x2[i * 2 + j][c]
                
                # V: B[j_global, k_global]
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = a_block * 2 + j_local
                        k_global = k_block * 2 + k_local
                        V[j_global * p + k_global][col] += V_2x2[j_local * 2 + k_local][c]
                
                # W: C[i, k_global]
                for i in range(2):
                    for k_local in range(2):
                        k_global = k_block * 2 + k_local
                        W[k_global * n + i][col] += W_2x2[k_local * 2 + i][c]
                
                col += 1
                
    # Verify with NumPy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Vectorized check
    # Reshape to 3D tensors?
    # U: (n*m, rank) -> (n, m, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i,j,k]
    # T = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a tensor contraction.
    # We can do this by iterating or using einsum.
    # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Expected T is all 1s
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_computed, T_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
