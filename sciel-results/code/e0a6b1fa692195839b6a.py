"""Optimized tensor decomposition for 2x3 x 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-22 decomposition for 2x3 x 3x4 matrix multiplication.
    
    Split B's columns into two 2-column blocks:
    C[:, 0:2] = A @ B[:, 0:2]  (2x3 x 3x2)
    C[:, 2:4] = A @ B[:, 2:4]  (2x3 x 3x2)
    
    Each 2x3 x 3x2 is split as:
    A[:, 0:2] @ B[0:2, :] + A[:, 2:3] @ B[2:3, :]
    = (2x2 x 2x2) + (2x1 x 1x2)
    = rank 7 (Strassen) + rank 4
    = rank 11
    
    Total rank: 22
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]   # 6 x 22
    V = [[0.0] * rank for _ in range(m * p)]   # 12 x 22
    W = [[0.0] * rank for _ in range(p * n)]   # 8 x 22
    
    # We'll build the decomposition in blocks.
    # Block 1: C[:, 0:2] from A @ B[:, 0:2]
    # Block 2: C[:, 2:4] from A @ B[:, 2:4]
    
    def strassen_2x2_x_2x2(U_offset, V_offset, W_offset, a_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen 2x2 x 2x2 decomposition.
        a_rows: which 2 rows of A (indices)
        b_cols: which 2 cols of B (indices)
        c_rows: which 2 rows of C (indices)
        c_cols: which 2 cols of C (indices)
        """
        # Strassen factors for 2x2 x 2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # U index: a_rows[0]*3 + 0, a_rows[0]*3 + 1, a_rows[1]*3 + 0, a_rows[1]*3 + 1
        # V index: 0*4 + b_cols[0], 0*4 + b_cols[1], 1*4 + b_cols[0], 1*4 + b_cols[1]
        # W index: (col)*2 + row  (trace convention W[k*n+i, r])
        
        a00 = a_rows[0] * m + 0
        a01 = a_rows[0] * m + 1
        a10 = a_rows[1] * m + 0
        a11 = a_rows[1] * m + 1
        
        b00 = 0 * p + b_cols[0]
        b01 = 0 * p + b_cols[1]
        b10 = 1 * p + b_cols[0]
        b11 = 1 * p + b_cols[1]
        
        w00 = b_cols[0] * n + c_rows[0]
        w01 = b_cols[1] * n + c_rows[0]
        w10 = b_cols[0] * n + c_rows[1]
        w11 = b_cols[1] * n + c_rows[1]
        
        # M1 = (A00+A11)(B00+B11)
        U[a00][U_offset] = 1
        U[a11][U_offset] = 1
        V[b00][V_offset] = 1
        V[b11][V_offset] = 1
        W[w00][W_offset] = 1
        W[w01][W_offset] = 1
        W[w10][W_offset] = 1
        W[w11][W_offset] = 1
        
        # M2 = (A10+A11)B00
        U[a10][U_offset + 1] = 1
        U[a11][U_offset + 1] = 1
        V[b00][V_offset + 1] = 1
        W[w10][W_offset + 1] = 1
        W[w11][W_offset + 1] = 1
        
        # M3 = A00(B10-B11)
        U[a00][U_offset + 2] = 1
        V[b10][V_offset + 2] = 1
        V[b11][V_offset + 2] = -1
        W[w01][W_offset + 2] = 1
        W[w11][W_offset + 2] = 1
        
        # M4 = A11(B01-B10)
        U[a11][U_offset + 3] = 1
        V[b01][V_offset + 3] = 1
        V[b10][V_offset + 3] = -1
        W[w00][W_offset + 3] = 1
        W[w10][W_offset + 3] = 1
        
        # M5 = (A00+A01)B11
        U[a00][U_offset + 4] = 1
        U[a01][U_offset + 4] = 1
        V[b11][V_offset + 4] = 1
        W[w00][W_offset + 4] = -1
        W[w10][W_offset + 4] = -1
        
        # M6 = (A10-A00)(B00+B10)
        U[a10][U_offset + 5] = 1
        U[a00][U_offset + 5] = -1
        V[b00][V_offset + 5] = 1
        V[b10][V_offset + 5] = 1
        W[w11][W_offset + 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[a01][U_offset + 6] = 1
        U[a11][U_offset + 6] = -1
        V[b01][V_offset + 6] = 1
        V[b11][V_offset + 6] = 1
        W[w00][W_offset + 6] = 1
        W[w11][W_offset + 6] = 1
        
        return 7  # 7 factors used
    
    def linear_2x1_x_1x2(U_offset, V_offset, W_offset, a_rows, b_row, c_rows, c_cols):
        """
        Add 2x1 x 1x2 decomposition (4 multiplications).
        a_rows: 2 rows of A
        b_row: 1 row of B
        c_rows: 2 rows of C
        c_cols: 2 cols of C
        """
        a0 = a_rows[0] * m + 2  # column 2 of A
        a1 = a_rows[1] * m + 2
        
        b0 = b_row * p + c_cols[0]
        b1 = b_row * p + c_cols[1]
        
        w00 = c_cols[0] * n + c_rows[0]
        w01 = c_cols[1] * n + c_rows[0]
        w10 = c_cols[0] * n + c_rows[1]
        w11 = c_cols[1] * n + c_rows[1]
        
        # C00 = A0 * B0
        U[a0][U_offset] = 1
        V[b0][V_offset] = 1
        W[w00][W_offset] = 1
        
        # C10 = A1 * B0
        U[a1][U_offset + 1] = 1
        V[b0][V_offset + 1] = 1
        W[w10][W_offset + 1] = 1
        
        # C01 = A0 * B1
        U[a0][U_offset + 2] = 1
        V[b1][V_offset + 2] = 1
        W[w01][W_offset + 2] = 1
        
        # C11 = A1 * B1
        U[a1][U_offset + 3] = 1
        V[b1][V_offset + 3] = 1
        W[w11][W_offset + 3] = 1
        
        return 4  # 4 factors used
    
    # Block 1: C[:, 0:2] = A @ B[:, 0:2]
    # = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:3] @ B[2:3, 0:2]
    col_offset = 0
    col_offset += strassen_2x2_x_2x2(
        U_offset=col_offset, V_offset=col_offset, W_offset=col_offset,
        a_rows=[0, 1], b_cols=[0, 1], c_rows=[0, 1], c_cols=[0, 1]
    )
    col_offset += linear_2x1_x_1x2(
        U_offset=col_offset, V_offset=col_offset, W_offset=col_offset,
        a_rows=[0, 1], b_row=2, c_rows=[0, 1], c_cols=[0, 1]
    )
    
    # Block 2: C[:, 2:4] = A @ B[:, 2:4]
    # = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:3] @ B[2:3, 2:4]
    col_offset += strassen_2x2_x_2x2(
        U_offset=col_offset, V_offset=col_offset, W_offset=col_offset,
        a_rows=[0, 1], b_cols=[2, 3], c_rows=[0, 1], c_cols=[2, 3]
    )
    col_offset += linear_2x1_x_1x2(
        U_offset=col_offset, V_offset=col_offset, W_offset=col_offset,
        a_rows=[0, 1], b_row=2, c_rows=[0, 1], c_cols=[2, 3]
    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

