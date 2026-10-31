import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return before the 120s limit
    # The construction is O(1) with small constants, so this is just a safety net.
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    # The known minimal rank for 4x4 matrix multiplication is 49.
    # This is achieved by the recursive application of Strassen's algorithm (2x2 blocks).
    rank = 49

    # Initialize matrices
    # U: (n*m) x rank = 16 x 49
    # V: (m*p) x rank = 16 x 49
    # W: (p*n) x rank = 16 x 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen coefficients for 2x2 matrix multiplication
    # We define the 7 multiplications M0..M6
    # For each multiplication s, we have:
    #   A_form: 2x2 matrix of coefficients for A
    #   B_form: 2x2 matrix of coefficients for B
    #   C_form: 2x2 matrix of coefficients for C
    
    # Standard Strassen:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # C00 = M0 - M3 + M4
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M6

    # Coefficients indexed by [s, i, j]
    # s in 0..6, i,j in 0..1
    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    # s=4
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    # s=6
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Construct U, V, W for 4x4x4
    # The 4x4 matrix A is viewed as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # Index mapping:
    # A[i, j] where i = 2*I + i2, j = 2*J + j2
    # I, J in {0, 1} are block indices
    # i2, j2 in {0, 1} are intra-block indices
    
    # The decomposition is a tensor product of the outer 2x2 Strassen and inner 2x2 Strassen.
    # Total rank = 7 (outer) * 7 (inner) = 49.
    # The rank index r = 7 * s_outer + t_inner
    
    # For U:
    # U[row, r] corresponds to the coefficient of A[i, j] in the r-th multiplication.
    # The r-th multiplication is defined by outer term s and inner term t.
    # The A-linear form for the r-th term is:
    # sum_{I, J, i2, j2} A_coeff[s, I, J] * A_coeff[t, i2, j2] * A[2*I+i2, 2*J+j2]
    # So U[2*I+i2, 2*J+j2, 7*s+t] = A_coeff[s, I, J] * A_coeff[t, i2, j2]
    
    # Note: U shape is (n*m, rank) = (16, 49).
    # Row index in U is i*m + j = i*4 + j.
    # i = 2*I + i2, j = 2*J + j2.
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    coeff_outer_A = A_coeff[s, I, J]
                    if coeff_outer_A == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            coeff_inner_A = A_coeff[t, i2, j2]
                            if coeff_inner_A == 0:
                                continue
                            val = coeff_outer_A * coeff_inner_A
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j, r] = val

            # Fill V
            # V[row, r] corresponds to coefficient of B[j, k].
            # B[j, k] where j = 2*J + j2, k = 2*K + k2.
            # V-linear form: sum_{J, K, j2, k2} B_coeff[s, J, K] * B_coeff[t, j2, k2] * B[2*J+j2, 2*K+k2]
            # V[2*J+j2, 2*K+k2, r] = B_coeff[s, J, K] * B_coeff[t, j2, k2]
            # Row index in V is j*p + k = j*4 + k.
            
            for J in range(2):
                for K in range(2):
                    coeff_outer_B = B_coeff[s, J, K]
                    if coeff_outer_B == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            coeff_inner_B = B_coeff[t, j2, k2]
                            if coeff_inner_B == 0:
                                continue
                            val = coeff_outer_B * coeff_inner_B
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k, r] = val

            # Fill W
            # W[row, r] corresponds to coefficient of C[i, k].
            # Convention: W[k*n + i, r].
            # C[i, k] where i = 2*I + i2, k = 2*K + k2.
            # W-linear form: sum_{I, K, i2, k2} C_coeff[s, I, K] * C_coeff[t, i2, k2] * C[2*I+i2, 2*K+k2]
            # W[2*K+k2, 2*I+i2, r] = C_coeff[s, I, K] * C_coeff[t, i2, k2]
            # Row index in W is k*n + i = k*4 + i.
            
            for I in range(2):
                for K in range(2):
                    coeff_outer_C = C_coeff[s, I, K]
                    if coeff_outer_C == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            coeff_inner_C = C_coeff[t, i2, k2]
                            if coeff_inner_C == 0:
                                continue
                            val = coeff_outer_C * coeff_inner_C
                            i = 2 * I + i2
                            k = 2 * K + k2
                            # Index is k*4 + i
                            W[k * 4 + i, r] = val

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
