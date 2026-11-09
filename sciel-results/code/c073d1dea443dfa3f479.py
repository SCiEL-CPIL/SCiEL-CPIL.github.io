"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Kronecker product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 27)
    # Total rank = 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # u_s, v_s, w_s are 2x2 matrices (flattened to 4 elements)
    # Indices: (i1, j1) for u, (j1, k1) for v, (k1, i1) for w
    # i1, j1, k1 in {0, 1}
    
    strassen_u = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[0, 0], [1, -1]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    strassen_v = [
        [[1, 0], [0, 1]], # B00 + B11
        [[1, 0], [0, 0]], # B00
        [[0, 1], [0, -1]],# B01 - B11
        [[0, -1], [0, 1]],# B10 - B00
        [[0, 0], [0, 1]], # B11
        [[1, 1], [0, 0]], # B00 + B01
        [[0, 1], [1, 1]]  # B10 + B11
    ]
    
    strassen_w = [
        [[1, 0], [0, 1]], # C00 += M1, C11 += M1
        [[0, 0], [1, 0]], # C10 += M2
        [[0, 1], [0, 0]], # C01 += M3
        [[0, -1], [1, 0]],# C10 += M4, C00 -= M4
        [[0, 1], [0, 0]], # C01 += M5
        [[0, 0], [0, 1]], # C11 += M6
        [[0, 1], [-1, 0]] # C01 += M7, C00 += M7
    ]
    
    # Note: w indices are (k1, i1)
    # w[k1, i1]
    
    # Schoolbook for 3x3
    # p_t, q_t, r_t are 3x3 matrices
    # t corresponds to (i2, j2, k2)
    
    # We iterate over all 7 Strassen steps and 27 Schoolbook steps
    col = 0
    for s in range(7):
        for i2 in range(3):
            for j2 in range(3):
                for k2 in range(3):
                    # This is the t-th term for this s
                    # p_t(i2, j2) = 1
                    # q_t(j2, k2) = 1
                    # r_t(k2, i2) = 1
                    
                    # U contribution
                    # U[i, j] where i = i1*3 + i2, j = j1*3 + j2
                    # U[i, j] = strassen_u[s][i1][j1] * 1
                    for i1 in range(2):
                        for j1 in range(2):
                            i = i1 * 3 + i2
                            j = j1 * 3 + j2
                            U[i][col] = strassen_u[s][i1][j1]
                            
                    # V contribution
                    # V[j, k] where j = j1*3 + j2, k = k1*3 + k2
                    # V[j, k] = strassen_v[s][j1][k1] * 1
                    for j1 in range(2):
                        for k1 in range(2):
                            j = j1 * 3 + j2
                            k = k1 * 3 + k2
                            V[j][col] = strassen_v[s][j1][k1]
                            
                    # W contribution
                    # W[k, i] where k = k1*3 + k2, i = i1*3 + i2
                    # W[k, i] = strassen_w[s][k1][i1] * 1
                    for k1 in range(2):
                        for i1 in range(2):
                            k = k1 * 3 + k2
                            i = i1 * 3 + i2
                            W[k][col] = strassen_w[s][k1][i1]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
