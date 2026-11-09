"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block splits
    # A: 3x5 -> rows [0], [1,2]; cols [0,1], [2,3,4]
    # B: 5x7 -> rows [0,1], [2,3,4]; cols [0,1,2], [3,4,5,6]
    
    # Total rank calculation
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_standard(n_loc, m_loc, p_loc, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1

    def add_strassen(i_off, j_off, k_off):
        # 2x2x2 Strassen
        nonlocal col
        # U, V, W matrices for 7 steps
        steps = [
            # U, V, W
            ([[1, 0], [0, 1]], [[1, 1], [1, 1]], [[1, 0], [0, 1]]),
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
            ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 1]]),
            ([[0, 0], [0, 1]], [[-1, 1], [0, 0]], [[1, 0], [1, 0]]),
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
            ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])
        ]
        
        for U_step, V_step, W_step in steps:
            for i in range(2):
                for j in range(2):
                    val_u = U_step[i][j]
                    if val_u != 0:
                        U[(i_off + i) * m + (j_off + j)][col] = val_u
            for j in range(2):
                for k in range(2):
                    val_v = V_step[j][k]
                    if val_v != 0:
                        V[(j_off + j) * p + (k_off + k)][col] = val_v
            for k in range(2):
                for i in range(2):
                    val_w = W_step[k][i]
                    if val_w != 0:
                        W[(k_off + k) * n + (i_off + i)][col] = val_w
            col += 1

    # 1. A11 B11: 1x2 by 2x3. Rank 6.
    # A11: rows 0, cols 0,1. B11: rows 0,1, cols 0,1,2.
    add_standard(1, 2, 3, 0, 0, 0)
    
    # 2. A12 B21: 1x3 by 3x3. Rank 9.
    # A12: rows 0, cols 2,3,4. B21: rows 2,3,4, cols 0,1,2.
    add_standard(1, 3, 3, 0, 2, 2)
    
    # 3. A11 B12: 1x2 by 2x4. Rank 8.
    # A11: rows 0, cols 0,1. B12: rows 0,1, cols 3,4,5,6.
    add_standard(1, 2, 4, 0, 0, 3)
    
    # 4. A12 B22: 1x3 by 3x4. Rank 12.
    # A12: rows 0, cols 2,3,4. B22: rows 2,3,4, cols 3,4,5,6.
    add_standard(1, 3, 4, 0, 2, 3)
    
    # 5. A21 B11: 2x2 by 2x3. Rank 11.
    # A21: rows 1,2, cols 0,1. B11: rows 0,1, cols 0,1,2.
    # Split B11 into B11a (2x2) and B11b (2x1).
    # A21 B11a: 2x2 by 2x2. Strassen.
    add_strassen(1, 0, 0)
    # A21 B11b: 2x2 by 2x1. Standard.
    add_standard(2, 2, 1, 1, 0, 2)
    
    # 6. A22 B21: 2x3 by 3x3. Rank 17.
    # A22: rows 1,2, cols 2,3,4. B21: rows 2,3,4, cols 0,1,2.
    # Split A22 into A22a (2x2) and A22b (2x1).
    # A22a B21: 2x2 by 3x3.
    # Split B21 into B21a (2x3) and B21b (1x3).
    # A22a B21a: 2x2 by 2x3. Rank 11.
    add_strassen(1, 2, 2) # A22a (rows 1,2; cols 2,3) x B21a (rows 2,3; cols 0,1,2)
    # Wait, B21a is 2x3. Strassen is 2x2.
    # We need 2x2 by 2x3.
    # We already handled 2x2 by 2x3 in step 5 logic.
    # Let's reuse the pattern: Strassen on first 2 cols of B, Standard on last 1 col.
    # A22a is rows 1,2; cols 2,3.
    # B21a is rows 2,3; cols 0,1,2.
    # Add Strassen for cols 0,1 of B21a.
    add_strassen(1, 2, 2)
    # Add Standard for col 2 of B21a.
    add_standard(2, 2, 1, 1, 2, 4) # A22a (rows 1,2; cols 2,3) x B21_col2 (row 2,3? No, B21 row 2,3 is 2 rows. Col 2 is 1 col.)
    # Wait, B21 is 3x3. Rows 2,3,4.
    # B21a is rows 2,3 (2 rows). Cols 0,1,2 (3 cols).
    # A22a is 2x2.
    # A22a * B21a (2x2 by 2x3).
    # Strassen on first 2 cols: add_strassen(1, 2, 2) -> covers B cols 0,1.
    # Standard on last 1 col: add_standard(2, 2, 1, 1, 2, 4) -> A rows 1,2; A cols 2,3; B rows 2,3; B col 4? No, B col 2.
    # B col index is 2. So k_off = 2.
    add_standard(2, 2, 1, 1, 2, 2)
    
    # A22b B21b: 2x1 by 1x3. Rank 6.
    # A22b: rows 1,2; col 4. B21b: row 4; cols 0,1,2.
    add_standard(2, 1, 3, 1, 4, 4)
    
    # 7. A21 B12: 2x2 by 2x4. Rank 14.
    # A21: rows 1,2; cols 0,1. B12: rows 0,1; cols 3,4,5,6.
    # Split B12 into two 2x2 blocks.
    add_strassen(1, 0, 3) # B cols 3,4
    add_strassen(1, 0, 5) # B cols 5,6
    
    # 8. A22 B22: 2x3 by 3x4. Rank 22.
    # A22: rows 1,2; cols 2,3,4. B22: rows 2,3,4; cols 3,4,5,6.
    # Split A22 into A22a (2x2) and A22b (2x1).
    # A22a B22: 2x2 by 3x4.
    # Split B22 into B22a (2x4) and B22b (1x4).
    # A22a B22a: 2x2 by 2x4. Rank 14.
    add_strassen(1, 2, 3) # B cols 3,4
    add_strassen(1, 2, 5) # B cols 5,6
    # A22b B22b: 2x1 by 1x4. Rank 8.
    add_standard(2, 1, 4, 1, 4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
