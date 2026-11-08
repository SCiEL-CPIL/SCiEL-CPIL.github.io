"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Partition A (2x4) into 2 blocks of 2x2 along columns
    # Partition B (4x6) into 3 blocks of 2x2 along columns
    # Each block multiplication is 2x2 * 2x2 -> 2x2, done via Strassen (rank 7)
    # Total rank = 2 * 3 * 7 = 42
    
    num_a_blocks = m // 2  # 2
    num_b_blocks = p // 2  # 3
    strassen_rank = 7
    rank = num_a_blocks * num_b_blocks * strassen_rank  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for the first matrix (A block, 2x2)
    # A_block indices: (0,0), (0,1), (1,0), (1,1)
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    U_strassen = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(1, 0): 1, (1, 1): 1},   # M2
        {(0, 0): 1},               # M3
        {(1, 1): 1},               # M4
        {(0, 0): 1, (0, 1): 1},   # M5
        {(1, 0): 1, (0, 0): -1},  # M6
        {(0, 1): 1, (1, 1): -1}   # M7
    ]
    
    # Strassen coefficients for the second matrix (B block, 2x2)
    # B_block indices: (0,0), (0,1), (1,0), (1,1)
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B10
    # M7: B01 + B11
    V_strassen = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(0, 0): 1},               # M2
        {(0, 1): 1, (1, 1): -1},  # M3
        {(1, 0): 1, (0, 0): -1},  # M4
        {(1, 1): 1},               # M5
        {(0, 0): 1, (1, 0): 1},   # M6
        {(0, 1): 1, (1, 1): 1}    # M7
    ]
    
    # Output contributions for each Strassen term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(1, 0): 1, (1, 1): -1},  # M2
        {(0, 1): 1, (1, 1): 1},   # M3
        {(0, 0): 1, (1, 0): 1},   # M4
        {(0, 0): -1, (0, 1): 1},  # M5
        {(1, 1): 1},               # M6
        {(0, 0): 1}                # M7
    ]
    
    col = 0
    for a_block in range(num_a_blocks):  # A column block
        for b_block in range(num_b_blocks):  # B column block
            for s in range(strassen_rank):  # Strassen term
                # Update U: A[i, a_block*2 + j]
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    U[i * m + j][col] += coeff
                
                # Update V: B[a_block*2 + j, b_block*2 + k]
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    V[j * p + k][col] += coeff
                
                # Update W: C[b_block*2 + k, i]
                for (i_loc, k_loc), coeff in W_strassen[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    W[k * n + i][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
