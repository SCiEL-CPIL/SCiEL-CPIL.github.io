"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    # k: 0..6
    # beta_X[k][I][J] : coefficient of A[I][J] in X_k
    # beta_Y[k][I][J] : coefficient of B[I][J] in Y_k
    # alpha[k][I][J]   : coefficient of M_k in C[I][J]
    
    # Using 0-based block indices: 00, 01, 10, 11
    # A00, A01, A10, A11
    
    beta_X = [
        [[1, 0], [0, 1]],  # M0: A00 + A11
        [[0, 0], [1, 1]],  # M1: A10 + A11
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: A00 + A01
        [[-1, 0], [1, 0]], # M5: A10 - A00
        [[0, 1], [0, -1]]  # M6: A01 - A11
    ]
    
    beta_Y = [
        [[1, 0], [0, 1]],  # M0: B00 + B11
        [[1, 0], [0, 0]],  # M1: B00
        [[-1, 0], [0, 1]], # M2: B10 - B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: B00 + B01
        [[0, 0], [1, 1]]   # M6: B10 + B11
    ]
    
    alpha = [
        [[1, 0], [0, 1]],  # M0: C00 + C11
        [[0, 0], [1, -1]], # M1: C10 - C11
        [[0, 1], [0, 1]],  # M2: C01 + C11
        [[1, 0], [1, 0]],  # M3: C00 + C10
        [[-1, 1], [0, 0]], # M4: -C00 + C01
        [[0, 0], [0, 1]],  # M5: C11
        [[1, 0], [0, 0]]   # M6: C00
    ]
    
    for k in range(7):
        for u in range(3):
            for w in range(3):
                for v in range(3):
                    r = k * 27 + u * 9 + w * 3 + v
                    
                    # Fill U
                    for I in range(2):
                        for J in range(2):
                            coeff = beta_X[k][I][J]
                            if coeff != 0:
                                i = 3 * I + u
                                j = 3 * J + w
                                a_idx = i * 6 + j
                                U[a_idx][r] = coeff
                                
                    # Fill V
                    for I in range(2):
                        for J in range(2):
                            coeff = beta_Y[k][I][J]
                            if coeff != 0:
                                i = 3 * I + w
                                j = 3 * J + v
                                b_idx = i * 6 + j
                                V[b_idx][r] = coeff
                                
                    # Fill W
                    for I in range(2):
                        for J in range(2):
                            coeff = alpha[k][I][J]
                            if coeff != 0:
                                i = 3 * I + u
                                j = 3 * J + v
                                c_idx = i * 6 + j
                                W[c_idx][r] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
