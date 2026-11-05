"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # Rank = 7 * (3*3*3) = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # alpha[s][ib][jb] : coeff of A[ib][jb] in M[s]
    # beta[s][jb][kb]  : coeff of B[jb][kb] in M[s]
    # gamma[s][ic][kc] : coeff of M[s] in C[ic][kc]
    
    alpha = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 1], [1, 1]]   # M7
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]],  # M3
        [[1, 0], [1, 0]],  # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 0]]   # M7
    ]
    
    column = 0
    for s in range(7):
        for i_l in range(3):
            for j_l in range(3):
                for k_l in range(3):
                    # Local indices within 3x3 block
                    # Global indices
                    # A index: (ib*3 + il)*6 + (jb*3 + jl)
                    # B index: (jb*3 + jl)*6 + (kb*3 + kl)
                    # W index: (kb*3 + kl)*6 + (ib*3 + il) -> Wait, W[k*n+i]
                    # k_global = kb*3 + kl, i_global = ib*3 + il
                    # W index = k_global * 6 + i_global
                    
                    for ib in range(2):
                        for jb in range(2):
                            a_coeff = alpha[s][ib][jb]
                            if a_coeff != 0:
                                a_idx = (ib * 3 + i_l) * 6 + (jb * 3 + j_l)
                                U[a_idx][column] += a_coeff
                    
                    for jb in range(2):
                        for kb in range(2):
                            b_coeff = beta[s][jb][kb]
                            if b_coeff != 0:
                                b_idx = (jb * 3 + j_l) * 6 + (kb * 3 + k_l)
                                V[b_idx][column] += b_coeff
                    
                    for ic in range(2):
                        for kc in range(2):
                            c_coeff = gamma[s][ic][kc]
                            if c_coeff != 0:
                                # W index: k_global * n + i_global
                                # k_global = kc * 3 + k_l
                                # i_global = ic * 3 + i_l
                                w_idx = (kc * 3 + k_l) * 6 + (ic * 3 + i_l)
                                W[w_idx][column] += c_coeff
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
