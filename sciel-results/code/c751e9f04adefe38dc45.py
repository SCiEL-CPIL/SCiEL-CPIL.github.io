import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen coefficients for 2x2 matrix multiplication
    A_coeffs = [
        [[1, 0], [0, 1]],  # s=0: A00 + A11
        [[0, 0], [1, 1]],  # s=1: A10 + A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00 + A01
        [[-1, 0], [1, 0]], # s=5: A10 - A00
        [[0, 1], [0, -1]], # s=6: A01 - A11
    ]

    B_coeffs = [
        [[1, 0], [0, 1]],  # s=0: B00 + B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01 - B11
        [[-1, 0], [1, 0]], # s=3: B10 - B00
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00 + B01
        [[0, 0], [1, 1]],  # s=6: B10 + B11
    ]

    C_coeffs = [
        [[1, 0], [0, 1]],  # s=0: C00 += M, C11 += M
        [[0, 0], [1, -1]], # s=1: C10 += M, C11 -= M
        [[0, 1], [0, 1]],  # s=2: C01 += M, C11 += M
        [[1, 0], [1, 0]],  # s=3: C00 += M, C10 += M
        [[-1, 1], [0, 0]], # s=4: C00 -= M, C01 += M
        [[0, 0], [0, 1]],  # s=5: C11 += M
        [[1, 0], [0, 0]],  # s=6: C00 += M
    ]

    U = np.zeros((16, rank), dtype=np.int32)
    V = np.zeros((16, rank), dtype=np.int32)
    W = np.zeros((16, rank), dtype=np.int32)

    # Pre-compute non-zero entries for each coefficient matrix to speed up construction
    A_nz = []
    for s in range(7):
        nz = []
        for I in range(2):
            for J in range(2):
                if A_coeffs[s][I][J] != 0:
                    nz.append((I, J, A_coeffs[s][I][J]))
        A_nz.append(nz)
    
    B_nz = []
    for s in range(7):
        nz = []
        for J in range(2):
            for K in range(2):
                if B_coeffs[s][J][K] != 0:
                    nz.append((J, K, B_coeffs[s][J][K]))
        B_nz.append(nz)
    
    C_nz = []
    for s in range(7):
        nz = []
        for I in range(2):
            for K in range(2):
                if C_coeffs[s][I][K] != 0:
                    nz.append((I, K, C_coeffs[s][I][K]))
        C_nz.append(nz)

    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            # U construction: U[i*4+j, r]
            for I, J, a_s in A_nz[s]:
                for i_prime in range(2):
                    for j_prime in range(2):
                        a_t = A_coeffs[t][i_prime][j_prime]
                        if a_t == 0:
                            continue
                        i = 2 * I + i_prime
                        j = 2 * J + j_prime
                        U[i * 4 + j, r] = a_s * a_t

            # V construction: V[j*4+k, r]
            for J, K, b_s in B_nz[s]:
                for j_prime in range(2):
                    for k_prime in range(2):
                        b_t = B_coeffs[t][j_prime][k_prime]
                        if b_t == 0:
                            continue
                        j = 2 * J + j_prime
                        k = 2 * K + k_prime
                        V[j * 4 + k, r] = b_s * b_t

            # W construction: W[k*4+i, r]
            for I, K, c_s in C_nz[s]:
                for i_prime in range(2):
                    for k_prime in range(2):
                        c_t = C_coeffs[t][i_prime][k_prime]
                        if c_t == 0:
                            continue
                        k = 2 * K + k_prime
                        i = 2 * I + i_prime
                        W[k * 4 + i, r] = c_s * c_t

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
