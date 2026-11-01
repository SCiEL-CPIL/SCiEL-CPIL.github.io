import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Seconds
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms defined as:
    # (a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff, g_coeff, h_coeff, c11_coeff, c12_coeff, c21_coeff, c22_coeff)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (c+d)e
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        # M3 = a(f-h)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        # M4 = d(g-e)
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (a+b)h
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (c-a)(e+f)
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (b-d)(g+h)
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    
    for block in range(num_blocks):
        if time.time() > deadline - 5:
            break
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # A part: U[i*m+j, col]
            U[0 * m + 0][col] = a_c  # A[0,0]
            U[0 * m + 1][col] = b_c  # A[0,1]
            U[1 * m + 0][col] = c_c  # A[1,0]
            U[1 * m + 1][col] = d_c  # A[1,1]
            
            # B part: V[j*p+k, col]
            # e = B[0, 2*block], f = B[0, 2*block+1], g = B[1, 2*block], h = B[1, 2*block+1]
            V[0 * p + (2 * block)][col] = e_c
            V[0 * p + (2 * block + 1)][col] = f_c
            V[1 * p + (2 * block)][col] = g_c
            V[1 * p + (2 * block + 1)][col] = h_c
            
            # C part: W[k*n+i, col]
            # C[0, 2*block] = c11, C[0, 2*block+1] = c12, C[1, 2*block] = c21, C[1, 2*block+1] = c22
            W[(2 * block) * n + 0][col] = c11_c
            W[(2 * block + 1) * n + 0][col] = c12_c
            W[(2 * block) * n + 1][col] = c21_c
            W[(2 * block + 1) * n + 1][col] = c22_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
