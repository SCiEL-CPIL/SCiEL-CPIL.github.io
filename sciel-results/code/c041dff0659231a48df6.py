import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)  # 16 x 49
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 49
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 49

    def get_U_index(row, col):
        return row * 4 + col

    def get_V_index(row, col):
        return row * 4 + col

    def get_W_index(k, i):
        return k * 4 + i  # trace convention: W[k*n+i, r]

    # Strassen 2x2 formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6

    strassen_2x2 = [
        # M1 = (a+d)(e+h) -> contributes to C11, C22
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (c+d)*e -> contributes to C21, C22
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a*(f-h) -> contributes to C12, C22
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = d*(g-e) -> contributes to C11, C21
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a+b)*h -> contributes to C11, C12
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (c-a)*(e+f) -> contributes to C22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (b-d)*(g+h) -> contributes to C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Block-level Strassen for 4x4:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
    # P2 = (A21+A22)B11 -> C21 += P2, C22 -= P2
    # P3 = A11(B12-B22) -> C12 += P3, C22 += P3
    # P4 = A22(B21-B11) -> C11 += P4, C21 += P4
    # P5 = (A11+A12)B22 -> C11 -= P5, C12 += P5
    # P6 = (A21-A11)(B11+B12) -> C22 += P6
    # P7 = (A12-A22)(B21+B22) -> C11 += P7

    block_strassen = [
        # P1 = (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3 = A11(B12-B22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4 = A22(B21-B11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5 = (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6 = (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For each block-level term t (0..6) and each scalar-level term s (0..6),
    # we get one rank-1 term with index r = t*7 + s

    for t in range(7):
        block_A, block_B, block_C = block_strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_2x2[s]
            r_idx = t * 7 + s

            # Build U contribution
            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U[get_U_index(a_row, a_col), r_idx] += block_coeff * scalar_coeff

            # Build V contribution
            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V[get_V_index(b_row, b_col), r_idx] += block_coeff * scalar_coeff

            # Build W contribution
            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr  # i index of C
                    c_col = 2 * cj + sc  # k index of C (column of C)
                    w_idx = get_W_index(c_col, c_row)  # W[k*4+i]
                    W[w_idx, r_idx] += block_coeff * scalar_coeff

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
