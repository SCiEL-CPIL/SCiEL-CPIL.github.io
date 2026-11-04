import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Rank 44 decomposition
    # Part 1: 1x4x4 (row 0 of A times B) - 16 terms
    # Part 2: 2x4x4 (rows 1,2 of A times B) using Strassen on 2x2 blocks - 28 terms
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    
    # Part 1: 1x4x4
    # C[0,k] = sum_j A[0,j] * B[j,k]
    for j in range(m):  # j = 0,1,2,3
        for k in range(p):  # k = 0,1,2,3
            U[0*m + j][col] = 1  # A[0,j]
            V[j*p + k][col] = 1  # B[j,k]
            W[k*n + 0][col] = 1  # C[0,k] -> W[k*3+0]
            col += 1
    
    # Part 2: 2x4x4 using Strassen
    # A1 = [A[1,0], A[1,1], A[1,2], A[1,3]]
    # A2 = [A[2,0], A[2,1], A[2,2], A[2,3]]
    # B = [[B00, B01], [B10, B11]] each 2x2
    # C1_left = A1_left * B_left + A1_right * B_bottom_left  -- no
    
    # Let me use the standard Strassen decomposition for 2x2x2 applied to blocks.
    # 
    # C[1:3, 0:2] = A[1:3, 0:2] * B[0:2, 0:2] + A[1:3, 2:4] * B[2:4, 0:2]
    # C[1:3, 2:4] = A[1:3, 0:2] * B[0:2, 2:4] + A[1:3, 2:4] * B[2:4, 2:4]
    #
    # So we have four 2x2x2 multiplications:
    # 1. A[1:3,0:2] * B[0:2,0:2]
    # 2. A[1:3,0:2] * B[0:2,2:4]
    # 3. A[1:3,2:4] * B[2:4,0:2]
    # 4. A[1:3,2:4] * B[2:4,2:4]
    
    # For each 2x2x2 multiplication, use Strassen (7 terms)
    # Strassen terms for M * N:
    # S1 = (M00+M11)*(N00+N11)
    # S2 = (M10+M11)*N00
    # S3 = M00*(N01-N11)
    # S4 = M11*(N10-N00)
    # S5 = (M00+M01)*N11
    # S6 = (M10-M00)*(N00+N10)
    # S7 = (M01-M11)*(N10+N11)
    # P00 = S1 - S4 + S5 + S7
    # P01 = S3 + S1
    # P10 = S2 + S4
    # P11 = S1 - S3 + S6 + S2
    
    def strassen_block(a_row1, a_row2, b_col1, b_col2, c_row1, c_row2, c_col1, c_col2, col_start):
        """
        Compute C[c_row1:c_row2+1, c_col1:c_col2+1] += A[a_row1:a_row2+1, :] * B[:, c_col1:c_col2+1]
        where A is 2x2, B is 2x2.
        
        a_row1, a_row2: rows of A (0-indexed, these are the actual row indices in the 3x4 matrix)
        b_col1, b_col2: rows of B (0-indexed, these are the actual row indices in the 4x4 matrix)
        c_row1, c_row2: rows of C
        c_col1, c_col2: cols of C
        """
        # A submatrix: A[a_row1, b_col1], A[a_row1, b_col2], A[a_row2, b_col1], A[a_row2, b_col2]
        # B submatrix: B[b_col1, c_col1], B[b_col1, c_col2], B[b_col2, c_col1], B[b_col2, c_col2]
        
        a00 = a_row1 * m + b_col1
        a01 = a_row1 * m + b_col2
        a10 = a_row2 * m + b_col1
        a11 = a_row2 * m + b_col2
        
        b00 = b_col1 * p + c_col1
        b01 = b_col1 * p + c_col2
        b10 = b_col2 * p + c_col1
        b11 = b_col2 * p + c_col2
        
        c00 = c_col1 * n + c_row1
        c01 = c_col2 * n + c_row1
        c10 = c_col1 * n + c_row2
        c11 = c_col2 * n + c_row2
        
        # S1 = (A00+A11)*(B00+B11)
        U[a00][col_start] = 1
        U[a11][col_start] = 1
        V[b00][col_start] = 1
        V[b11][col_start] = 1
        W[c00][col_start] = 1
        W[c01][col_start] = 1
        W[c10][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # S2 = (A10+A11)*B00
        U[a10][col_start] = 1
        U[a11][col_start] = 1
        V[b00][col_start] = 1
        W[c10][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # S3 = A00*(B01-B11)
        U[a00][col_start] = 1
        V[b01][col_start] = 1
        V[b11][col_start] = -1
        W[c01][col_start] = 1
        col_start += 1
        
        # S4 = A11*(B10-B00)
        U[a11][col_start] = 1
        V[b10][col_start] = 1
        V[b00][col_start] = -1
        W[c00][col_start] = -1
        W[c10][col_start] = 1
        col_start += 1
        
        # S5 = (A00+A01)*B11
        U[a00][col_start] = 1
        U[a01][col_start] = 1
        V[b11][col_start] = 1
        W[c00][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # S6 = (A10-A00)*(B00+B10)
        U[a10][col_start] = 1
        U[a00][col_start] = -1
        V[b00][col_start] = 1
        V[b10][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # S7 = (A01-A11)*(B10+N11)
        U[a01][col_start] = 1
        U[a11][col_start] = -1
        V[b10][col_start] = 1
        V[b11][col_start] = 1
        W[c01][col_start] = 1
        col_start += 1
        
        return col_start
    
    # Block 1: A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    col = strassen_block(1, 2, 0, 1, 1, 2, 0, 1, col)
    
    # Block 2: A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    col = strassen_block(1, 2, 0, 1, 1, 2, 2, 3, col)
    
    # Block 3: A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    col = strassen_block(1, 2, 2, 3, 1, 2, 0, 1, col)
    
    # Block 4: A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    col = strassen_block(1, 2, 2, 3, 1, 2, 2, 3, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
