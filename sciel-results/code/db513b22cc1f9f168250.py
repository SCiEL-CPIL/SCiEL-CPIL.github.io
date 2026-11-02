import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    
    # Strategy: 5 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    # Total rank = 5*7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrices
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-f)
        ({3: 1}, {2: 1, 1: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # 5 Strassen blocks
    for b in range(5):
        for u_dict, v_dict, w_dict in strassen_terms:
            # U is same for all blocks (depends only on A)
            for idx_a, val in u_dict.items():
                # idx_a is 0..3 corresponding to i*m+j
                # i = idx_a // 2, j = idx_a % 2
                U[idx_a][col] = val
            
            # V depends on block offset
            for idx_b, val in v_dict.items():
                # idx_b is local 0..3 corresponding to j*2+k_local
                # j = idx_b // 2, k_local = idx_b % 2
                j = idx_b // 2
                k_local = idx_b % 2
                k_global = 2 * b + k_local
                idx_v = j * p + k_global
                V[idx_v][col] = val
            
            # W depends on block offset
            for idx_c, val in w_dict.items():
                # idx_c is local 0..3 corresponding to k_local*2+i
                # k_local = idx_c // 2, i = idx_c % 2
                k_local = idx_c // 2
                i = idx_c % 2
                k_global = 2 * b + k_local
                idx_w = k_global * n + i
                W[idx_w][col] = val
            
            col += 1
    
    # 1 Schoolbook block for k=10
    # 2x2x1: 4 terms
    for i in range(n):
        for j in range(m):
            # A[i,j] * B[j, 10] -> C[i, 10]
            idx_u = i * m + j
            idx_v = j * p + 10
            idx_w = 10 * n + i
            U[idx_u][col] = 1
            V[idx_v][col] = 1
            W[idx_w][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
