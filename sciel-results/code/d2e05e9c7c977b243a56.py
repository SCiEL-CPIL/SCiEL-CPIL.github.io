import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2
    a_coeffs = [
        [[1, 0], [0, 1]],   # M0 = (A00 + A11)(B00 + B11)
        [[0, 0], [1, 1]],   # M1 = (A10 + A11)B00
        [[1, 0], [0, 0]],   # M2 = A00(B01 - B11)
        [[0, 0], [0, 1]],   # M3 = A11(B10 - B00)
        [[1, 1], [0, 0]],   # M4 = (A00 + A01)B11
        [[-1, 0], [1, 0]],  # M5 = (A10 - A00)(B00 + B01)
        [[0, 1], [0, -1]],  # M6 = (A01 - A11)(B10 + B11)
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # (B00+B11)
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # (B01-B11)
        [[-1, 0], [1, 0]],  # (B10-B00)
        [[0, 0], [0, 1]],   # B11
        [[1, 1], [0, 0]],   # (B00+B01)
        [[0, 0], [1, 1]],   # (B10+B11)
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

    # Build 49-rank decomposition via recursive Strassen
    # A is 4x4: A[i,j], i,j in 0..3
    # B is 4x4: B[j,k], j,k in 0..3
    # C is 4x4: C[i,k] = sum_j A[i,j]*B[j,k]
    # W follows trace convention: W[k*n+i, r] where n=4, so W[k*4+i, r]

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            # U[i*4+j, c_idx]: coefficient of A[i,j]
            # i = 2*S + s2, j = 2*T + t2
            # coeff = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            for S in range(2):
                for T in range(2):
                    a_outer = a_coeffs[s][S][T]
                    if a_outer == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_inner = a_coeffs[t][s2][t2]
                            if a_inner == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += a_outer * a_inner

            # V[j*4+k, c_idx]: coefficient of B[j,k]
            # j = 2*T + t2, k = 2*K + k2
            # coeff = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            for T in range(2):
                for K in range(2):
                    b_outer = b_coeffs[s][T][K]
                    if b_outer == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_inner = b_coeffs[t][t2][k2]
                            if b_inner == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += b_outer * b_inner

            # W[k*4+i, c_idx]: coefficient for C[i,k]
            # k = 2*K + k2, i = 2*S + s2
            # coeff = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            for K in range(2):
                for S in range(2):
                    c_outer = c_coeffs[s][S][K]
                    if c_outer == 0:
                        continue
                    for s2 in range(2):
                        for k2 in range(2):
                            c_inner = c_coeffs[t][s2][k2]
                            if c_inner == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][c_idx] += c_outer * c_inner

            if time.time() > deadline:
                return {"U": U, "V": W if False else W, "W": W}

    return {"U": U, "V": V, "W": W}
