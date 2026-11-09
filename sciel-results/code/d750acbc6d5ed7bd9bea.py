"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen (rank 7) for 2x2 blocks of 3x3 matrices
    # and Schoolbook (rank 27) for 3x3 matrices.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (X_s)
    # s is 0-indexed 0..6
    # Each entry is a list of (block_row, block_col, coeff)
    X_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],       # M2: A10 + A11
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],       # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],      # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)]       # M7: A01 - A11
    ]
    
    # Strassen coefficients for B (Y_s)
    Y_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1: B00 + B11
        [(0, 0, 1)],                  # M2: B00
        [(0, 1, 1), (1, 1, -1)],      # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],      # M4: B10 - B00
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (0, 1, 1)],       # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]        # M7: B10 + B11
    ]
    
    # Output coefficients for C
    # s is 0-indexed 0..6
    # Each entry is a list of (block_row, block_col, coeff)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],      # M2 -> C10, C11(-1)
        [(0, 1, 1), (1, 1, 1)],       # M3 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],       # M4 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01
        [(1, 1, 1)],                  # M6 -> C11
        [(0, 0, 1)]                   # M7 -> C00
    ]
    
    # Iterate over Strassen terms
    for s in range(7):
        # Iterate over Schoolbook terms for 3x3
        for t in range(27):
            col = s * 27 + t
            
            # Map t to (i, j, k) for 3x3 indices
            i = t // 9
            j = (t // 3) % 3
            k = t % 3
            
            # Fill U
            for br, bc, coeff in X_coeffs[s]:
                # A[br, bc] block, element (i, j)
                row_u = (br * 3 + i) * 6 + (bc * 3 + j)
                U[row_u][col] += coeff
                
            # Fill V
            for br, bc, coeff in Y_coeffs[s]:
                # B[br, bc] block, element (j, k)
                row_v = (br * 3 + j) * 6 + (bc * 3 + k)
                V[row_v][col] += coeff
                
            # Fill W
            for br, bc, coeff in W_coeffs[s]:
                # C[br, bc] block, element (i, k)
                # W index convention: k*n + i
                row_w = (bc * 3 + k) * 6 + (br * 3 + i)
                W[row_w][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
