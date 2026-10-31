import time
import numpy as np

def construct():
    # The rank of the 4x4 matrix multiplication tensor is known to be 49.
    # This is achieved by the recursive application of Strassen's algorithm.
    # Score is seed/rank. Assuming seed=64 (schoolbook), max score is 64/49.
    
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # Coefficients for A, B, C
    # a_coeffs[s][i][j] is coeff of A[i][j] in s-th Strassen input
    # b_coeffs[s][j][k] is coeff of B[j][k] in s-th Strassen input
    # c_coeffs[s][i][k] is coeff of C[i][k] from s-th Strassen output
    
    a_coeffs = [
        [[1, 0], [0, 1]],   # M0
        [[0, 0], [1, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 0], [0, 1]],   # M3
        [[1, 1], [0, 0]],   # M4
        [[-1, 0], [1, 0]],  # M5
        [[0, 1], [0, -1]],  # M6
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # M0
        [[1, 0], [0, 0]],   # M1
        [[0, 1], [0, -1]],  # M2
        [[-1, 0], [1, 0]],  # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[0, 0], [1, 1]],   # M6
    ]

    c_coeffs = [
        [[1, 0], [0, 1]],   # C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # C11+=M5
        [[1, 0], [0, 0]],   # C00+=M6
    ]

    # Recursive construction
    # Outer loop over 7 Strassen steps for 4x4 (composed of 2x2 blocks)
    # Inner loop over 7 Strassen steps for 2x2 blocks
    
    for s in range(7):
        for t in range(7):
            r_idx = 7 * s + t
            
            # U[i*4+j, r_idx]
            for S in range(2):
                for T in range(2):
                    a_outer = a_coeffs[s][S][T]
                    if a_outer == 0: continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_inner = a_coeffs[t][s2][t2]
                            if a_inner == 0: continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][r_idx] += a_outer * a_inner

            # V[j*4+k, r_idx]
            for T in range(2):
                for K in range(2):
                    b_outer = b_coeffs[s][T][K]
                    if b_outer == 0: continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_inner = b_coeffs[t][t2][k2]
                            if b_inner == 0: continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][r_idx] += b_outer * b_inner

            # W[k*4+i, r_idx]
            for K in range(2):
                for S in range(2):
                    c_outer = c_coeffs[s][S][K]
                    if c_outer == 0: continue
                    for s2 in range(2):
                        for k2 in range(2):
                            c_inner = c_coeffs[t][s2][k2]
                            if c_inner == 0: continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][r_idx] += c_outer * c_inner

    return {"U": U, "V": V, "W": W}
