import time

DIMENSIONS = [6, 6, 6]

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    # We use a blocked Strassen decomposition.
    # 6x6 matrix as 2x2 blocks of 3x3.
    # Strassen rank 7 for 2x2 blocks.
    # Schoolbook rank 27 for 3x3 blocks.
    # Total rank 7 * 27 = 189.
    
    n_blocks = 2
    base_n = 3
    rank = 7 * 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 blocks
    # Indices: A11 A12 A21 A22 -> A00 A01 A10 A11
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output coefficients S[l][R][C]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    S = [
        [[1, 0], [0, 1]], # M1 in C00, C11
        [[0, 0], [1, -1]], # M2 in C10, C11
        [[0, 1], [0, 1]], # M3 in C01, C11
        [[1, 0], [1, 0]], # M4 in C00, C10
        [[-1, 1], [0, 0]], # M5 in C00, C01
        [[0, 0], [0, 1]], # M6 in C11
        [[1, 0], [0, 0]], # M7 in C00
    ]
    
    # Strassen input A coefficients u[l][R][C]
    # M1: A00+ A11
    # M2: A10+ A11
    # M3: A00
    # M4: A11
    # M5: A00+ A01
    # M6: A10- A00
    # M7: A01- A11
    
    u = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, -1], [0, 1]], # M7
    ]
    
    # Strassen input B coefficients v[l][R][C]
    # M1: B00+ B11
    # M2: B00
    # M3: B01- B11
    # M4: B10- B00
    # M5: B11
    # M6: B00+ B01
    # M7: B10+ B11
    
    v = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[0, -1], [1, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [0, 1]], # M7
    ]
    
    col = 0
    for l in range(7):
        for il in range(3):
            for jl in range(3):
                for kl in range(3):
                    # Fill U: X_l[il, jl] = sum_{R,C} u[l][R][C] A[3R+il, 3C+jl]
                    for R in range(2):
                        for C in range(2):
                            coeff = u[l][R][C]
                            if coeff != 0:
                                i_glob = 3 * R + il
                                j_glob = 3 * C + jl
                                U[i_glob * 6 + j_glob][col] = coeff
                    
                    # Fill V: Y_l[jl, kl] = sum_{R,C} v[l][R][C] B[3R+jl, 3C+kl]
                    for R in range(2):
                        for C in range(2):
                            coeff = v[l][R][C]
                            if coeff != 0:
                                j_glob = 3 * R + jl
                                k_glob = 3 * C + kl
                                V[j_glob * 6 + k_glob][col] = coeff
                    
                    # Fill W: Contribution to Z_l[il, kl]
                    # Z_l is the product. The term corresponds to Z_l[il, kl].
                    # Z_l is added to C_blocks with weights S[l][R][C]
                    for R_out in range(2):
                        for C_out in range(2):
                            coeff = S[l][R_out][C_out]
                            if coeff != 0:
                                i_out_glob = 3 * R_out + il
                                k_out_glob = 3 * C_out + kl
                                # W index is k*n + i
                                W[k_out_glob * 6 + i_out_glob][col] = coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
