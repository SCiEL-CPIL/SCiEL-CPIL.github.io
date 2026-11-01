"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p // 2) since p is even (14)
    # If p is odd, we handle the last block separately or just use schoolbook for the remainder
    # For p=14, we have 7 blocks of 2x2.
    num_blocks = p // 2
    remainder = p % 2
    rank = 7 * num_blocks + (4 if remainder else 0)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M_t = sum_i A_i * sum_j B_j
    # We define the linear forms for A and B, and the combination for C
    
    # A coefficients for M1..M7 (indices i, j where A is i, j)
    # A is 2x2. Indices 0..3: 00, 01, 10, 11
    # M1: A00
    # M2: A00 + A10
    # M3: A01 + A11
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    A_coeffs = [
        {0: 1},               # M1
        {0: 1, 2: 1},         # M2 (A00, A10)
        {1: 1, 3: 1},         # M3 (A01, A11)
        {3: 1},               # M4 (A11)
        {0: 1, 1: 1},         # M5 (A00, A01)
        {2: 1, 0: -1},        # M6 (A10 - A00)
        {1: 1, 3: -1}         # M7 (A01 - A11)
    ]
    
    # B coefficients for M1..M7 (indices k, l where B is k, l? No, B is 2x2 in block)
    # B is 2x2. Indices 0..3: 00, 01, 10, 11
    # M1: B00 + B11
    # M2: B11
    # M3: B00
    # M4: B00 - B10
    # M5: B10
    # M6: B01 + B00
    # M7: B11 + B10
    
    B_coeffs = [
        {0: 1, 3: 1},         # M1 (B00, B11)
        {3: 1},               # M2 (B11)
        {0: 1},               # M3 (B00)
        {0: 1, 2: -1},        # M4 (B00 - B10)
        {2: 1},               # M5 (B10)
        {1: 1, 0: 1},         # M6 (B01 + B00)
        {3: 1, 2: 1}          # M7 (B11 + B10)
    ]
    
    # C coefficients for M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00 from M1, M4, M5, M7
        {1: 1, 4: 1},                # C01 from M3, M5
        {2: 1, 3: 1},                # C10 from M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11 from M1, M2, M3, M6
    ]
    
    for b in range(num_blocks):
        k_start = b * 2
        for t in range(7):
            # Update U
            for idx, coeff in A_coeffs[t].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][col] += coeff
                
            # Update V
            for idx, coeff in B_coeffs[t].items():
                k_local = idx // 2
                l_local = idx % 2
                # V index is j * p + k. Here j is the row of B (0 or 1), k is the column.
                # In Strassen, B is 2x2. Rows 0,1. Cols 0,1.
                # V[j, k_global, t]
                k_global = k_start + l_local
                # Wait, B_coeffs indices: 0:00, 1:01, 2:10, 3:11.
                # idx 0 -> row 0, col 0.
                # idx 1 -> row 0, col 1.
                # idx 2 -> row 1, col 0.
                # idx 3 -> row 1, col 1.
                row_B = idx // 2
                col_B = idx % 2
                k_global = k_start + col_B
                V[row_B * p + k_global][col] += coeff
                
            # Update W
            # W index is k * n + i.
            # C_coeffs maps output position (0:00, 1:01, 2:10, 3:11) to M_t contribution.
            for c_idx, coeff in C_coeffs[t].items():
                i_out = c_idx // 2
                k_out_local = c_idx % 2
                k_out_global = k_start + k_out_local
                W[k_out_global * n + i_out][col] += coeff
                
            col += 1
            
    # Handle remainder if p is odd (not the case here, but for robustness)
    if remainder:
        k_start = num_blocks * 2
        # Schoolbook for 2x1 block
        for i in range(n):
            for j in range(m):
                for k in range(remainder):
                    U[i * m + j][col] = 1
                    V[j * p + k_start + k][col] = 1
                    W[(k_start + k) * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
