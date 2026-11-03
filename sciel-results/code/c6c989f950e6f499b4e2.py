"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen on 2x2 blocks)
    # Rank 16 for 4x4x1 part (Schoolbook)
    # Total rank 65
    
    # Strassen coefficients for 2x2
    # a[alpha][I][J], b[alpha][I][J], c[alpha][K][L]
    # I, J, K, L in {0, 1}
    
    a_coeff = [
        [[1, 0], [0, 1]],  # 1: A00+A11
        [[0, 0], [1, 1]],  # 2: A10+A11
        [[1, 0], [0, 0]],  # 3: A00
        [[0, 0], [0, 1]],  # 4: A11
        [[1, 1], [0, 0]],  # 5: A00+A01
        [[-1, 0], [1, 0]], # 6: A10-A00
        [[0, 1], [0, -1]]  # 7: A01-A11
    ]
    
    b_coeff = [
        [[1, 0], [0, 1]],  # 1: B00+B11
        [[1, 0], [0, 0]],  # 2: B00
        [[0, 1], [0, -1]], # 3: B01-B11
        [[-1, 0], [1, 0]], # 4: B10-B00
        [[0, 0], [0, 1]],  # 5: B11
        [[1, 1], [0, 0]],  # 6: B00+B01
        [[0, 1], [1, 1]]   # 7: B10+B11
    ]
    
    c_coeff = [
        [[1, 0], [0, 1]],  # 1: C00+C11
        [[0, 0], [1, -1]], # 2: C10-C11
        [[0, 1], [0, 1]],  # 3: C01+C11
        [[1, 0], [1, 0]],  # 4: C00+C10
        [[-1, 1], [0, 0]], # 5: -C00+C01
        [[0, 0], [0, 1]],  # 6: C11
        [[1, 0], [0, 0]]   # 7: C00
    ]
    
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4x4x4 part
    for alpha in range(7):
        for beta in range(7):
            # Sub-Strassen coefficients for the block multiplication
            # We reuse the same a, b, c structures for the 2x2 sub-blocks
            # u_beta[x][y], v_beta[x][y], w_beta[y][z]
            u_beta = a_coeff[beta]
            v_beta = b_coeff[beta]
            w_beta = c_coeff[beta]
            
            for I in range(2):
                for J in range(2):
                    a_val = a_coeff[alpha][I][J]
                    if a_val == 0: continue
                    for x in range(2):
                        for y in range(2):
                            u_val = u_beta[x][y]
                            if u_val == 0: continue
                            
                            i = 2 * I + x
                            j = 2 * J + y
                            U[i * m + j][col] += a_val * u_val
                            
                            # V part: B is 4x4 for the first 4 cols
                            b_val = b_coeff[alpha][I][J]
                            if b_val != 0:
                                v_val = v_beta[x][y]
                                if v_val != 0:
                                    V[j * p + (2 * J + y)][col] += b_val * v_val
                                    
                            # W part: C is 4x4 for the first 4 cols
                            # C index k*n+i. k is col, i is row.
                            # C block K, L. Sub-index y, z.
                            # i = 2*L + y, k = 2*K + z.
                            # w_beta[y][z]
                            for K in range(2):
                                for L in range(2):
                                    c_val = c_coeff[alpha][K][L]
                                    if c_val == 0: continue
                                    for z in range(2):
                                        w_val = w_beta[y][z]
                                        if w_val == 0: continue
                                        i_out = 2 * L + y
                                        k_out = 2 * K + z
                                        W[k_out * n + i_out][col] += c_val * w_val
            col += 1
            
    # 4x4x1 part (5th column)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 16 terms
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
