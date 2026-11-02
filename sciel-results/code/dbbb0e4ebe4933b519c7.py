"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    
    # A coefficients (a11, a12, a21, a22)
    # B coefficients (b11, b12, b21, b22)
    # C coefficients (c11, c12, c21, c22)
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),   # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),  # M2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),  # M3
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),  # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),  # M5
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),  # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),  # M7
    ]
    
    # Define sub-problems
    # Each sub-problem is (A_block_indices, B_block_indices, C_block_indices)
    # A_block_indices: [a11, a12, a21, a22] global indices
    # B_block_indices: [b11, b12, b21, b22] global indices
    # C_block_indices: [c11, c12, c21, c22] global indices
    
    # A_L: rows 0-1, cols 0-1
    A_L = [0, 1, 4, 5]
    # A_R: rows 0-1, cols 2-3
    A_R = [2, 3, 6, 7]
    
    # B blocks (rows 0-1 are top, 2-3 are bottom)
    # B[row_start, col_start]
    def get_B_indices(row_start, col_start):
        # b11 = B[row_start, col_start]
        # b12 = B[row_start, col_start+1]
        # b21 = B[row_start+1, col_start]
        # b22 = B[row_start+1, col_start+1]
        idx = [
            row_start * 8 + col_start,
            row_start * 8 + col_start + 1,
            (row_start + 1) * 8 + col_start,
            (row_start + 1) * 8 + col_start + 1
        ]
        return idx

    # C blocks (cols 0-1, 2-3, etc)
    # C is 2x8. Trace convention W[k*n+i] where n=2.
    # C[i, k] -> index k*2 + i
    def get_C_indices(col_start):
        # c11 = C[0, col_start] -> col_start*2 + 0
        # c12 = C[0, col_start+1] -> (col_start+1)*2 + 0
        # c21 = C[1, col_start] -> col_start*2 + 1
        # c22 = C[1, col_start+1] -> (col_start+1)*2 + 1
        idx = [
            col_start * 2 + 0,
            (col_start + 1) * 2 + 0,
            col_start * 2 + 1,
            (col_start + 1) * 2 + 1
        ]
        return idx

    sub_problems = []
    # Top half of B (rows 0-1)
    for col_start in [0, 2, 4, 6]:
        sub_problems.append((A_L, get_B_indices(0, col_start), get_C_indices(col_start)))
    # Bottom half of B (rows 2-3)
    for col_start in [0, 2, 4, 6]:
        sub_problems.append((A_R, get_B_indices(2, col_start), get_C_indices(col_start)))
    
    col_idx = 0
    for A_idx, B_idx, C_idx in sub_problems:
        for A_c, B_c, C_c in strassen_terms:
            # Add to U
            for local_i, global_i in zip([0, 1, 2, 3], A_idx):
                U[global_i][col_idx] += A_c[local_i]
            # Add to V
            for local_i, global_i in zip([0, 1, 2, 3], B_idx):
                V[global_i][col_idx] += B_c[local_i]
            # Add to W
            for local_i, global_i in zip([0, 1, 2, 3], C_idx):
                W[global_i][col_idx] += C_c[local_i]
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
