import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen coefficients for 2x2 matrix multiplication
    # A: 2x2 matrix of coefficients for A[i,j]
    # B: 2x2 matrix of coefficients for B[j,k]
    # C: 2x2 matrix of coefficients for C[i,k]
    
    # Standard Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    A_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]], # M7
    ]

    B_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]],  # M7
    ]

    C_coeffs = [
        [[1, 0], [0, 1]],  # M1: C00+=M1, C11+=M1
        [[0, 0], [1, -1]], # M2: C10+=M2, C11-=M2
        [[0, 1], [0, 1]],  # M3: C01+=M3, C11+=M3
        [[1, 0], [1, 0]],  # M4: C00-=M4 (wait, C00 = M1 - M4 + ...), so C00+=-M4? 
                           # Let's re-verify C coefficients.
                           # C00 = M1 - M4 + M5 + M7
                           # C01 = M3 + M5
                           # C10 = M2 + M4
                           # C11 = M1 - M2 + M3 + M6
    ]
    
    # Correct C_coeffs:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 -= 1, C10 += 1
    # M5: C00 += 1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1

    C_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]],  # M3
        [[-1, 0], [1, 0]], # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 0]],  # M7
    ]

    # Pre-compute non-zero entries (index, value) for each coefficient matrix
    # U[i*4+j, r] where i,j are 0..3
    # r = 7*s + t
    # U[i*4+j, r] = A_s[i//2, j//2] * A_t[i%2, j%2]
    
    U = np.zeros((16, rank), dtype=np.int32)
    V = np.zeros((16, rank), dtype=np.int32)
    W = np.zeros((16, rank), dtype=np.int32)

    # Generate non-zero entries for A, B, C
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

    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # U
            for I, J, a_s in A_nz[s]:
                for i_prime in range(2):
                    for j_prime in range(2):
                        a_t = A_coeffs[t][i_prime][j_prime]
                        if a_t == 0:
                            continue
                        i = 2 * I + i_prime
                        j = 2 * J + j_prime
                        U[i * 4 + j, r] = a_s * a_t

            # V
            for J, K, b_s in B_nz[s]:
                for j_prime in range(2):
                    for k_prime in range(2):
                        b_t = B_coeffs[t][j_prime][k_prime]
                        if b_t == 0:
                            continue
                        j = 2 * J + j_prime
                        k = 2 * K + k_prime
                        V[j * 4 + k, r] = b_s * b_t

            # W
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
