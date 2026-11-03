import time

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Rank-32 tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Strategy: Split the 3x3 A matrix into 2x2 blocks and the 3x8 B matrix
    into 2x8 blocks, then apply Strassen's 7-product formula to each 2x2
    sub-block of B (treating B as 4 consecutive 2x2 blocks).
    
    Total rank = 4 blocks * 7 Strassen products = 28 for the 2x2 parts.
    The remaining 1x2 and 2x2 parts are handled with direct multiplications.
    
    Let's break it down:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x8, B12 is 2x8, B21 is 1x8, B22 is 1x8
    
    C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    For the 2x2 * 2x8 multiplications, we can use Strassen-like decomposition.
    Actually, let's use a simpler approach: treat each 2x2 * 2x8 as four 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 can be done with 7 multiplications (Strassen).
    So 2x2 * 2x8 = 4 * 7 = 28 multiplications.
    
    For 2x1 * 1x8: this is just 2*8 = 16 scalar multiplications (rank 16)
    For 1x2 * 2x8: this is just 16 scalar multiplications (rank 16)
    For 1x1 * 1x8: this is just 8 scalar multiplications (rank 8)
    
    Total = 28 (A11*B11) + 28 (A11*B12) + 16 (A12*B21) + 8 (A12*B22) + 16 (A21*B11) + 8 (A22*B21) + 16 (A21*B12) + 8 (A22*B22)
    = 28+28+16+8+16+8+16+8 = 128. That's worse than 72!
    
    Let me reconsider. The key insight is that we can share the Strassen decomposition across the 8 columns.
    
    Actually, for a 2x2 * 2x8 multiplication, we can view it as:
    Let A2x2 = [[a00, a01], [a10, a11]]
    Let B2x8 = [[b00, b01, b02, b03, b04, b05, b06, b07],
                [b10, b11, b12, b13, b14, b15, b16, b17]]
    
    C = A*B is 2x8.
    
    Using Strassen for each pair of columns doesn't work well. Instead, let's think of it differently.
    
    The correct approach: For 2x2 * 2x8, we can use 7 * 4 = 28 rank-1 tensors if we decompose B into 4 2x2 blocks.
    
    But wait, the original schoolbook is 3*3*8 = 72. We need to do better.
    
    Let me use the fact that 3 = 2 + 1.
    
    Actually, a known result: the rank of 3x3x3 matrix multiplication is at most 23 (Lader's decomposition).
    For 3x3x8, we can use 8 * 23 = 184? No, that's worse.
    
    Let me think again. The rank of M_{n,m,p} tensor.
    
    For 3x3 * 3x8:
    n=3, m=3, p=8
    
    One approach: use rank-23 decomposition for 3x3x3, and repeat for each group of 3 columns? But p=8 is not a multiple of 3.
    
    Alternative: Use the fact that we can embed 3x3x8 into larger structures.
    
    Let me try a different decomposition. Split the 3x3 A into:
    - A11: 2x2
    - A12: 2x1  
    - A21: 1x2
    - A22: 1x1
    
    And B (3x8) into:
    - B1: first 2 rows, all 8 cols (2x8)
    - B2: last row, all 8 cols (1x8)
    
    Then C = A*B where:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    For i=0,1 (first 2 rows of C):
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + A[i,2] B[2,k]
            = (A11 * B1)[i,k] + A12[i] * B2[k]
    
    For i=2 (last row of C):
    C[2,k] = sum_{j=0}^{1} A[2,j] B[j,k] + A[2,2] B[2,k]
            = (A21 * B1)[2,k] + A22 * B2[k]
    
    Now, A11 is 2x2, B1 is 2x8. The product A11*B1 is 2x8.
    A21 is 1x2, B1 is 2x8. The product A21*B1 is 1x8.
    
    For the 2x2 * 2x8 part: We can use 7 Strassen-like products for each 2x2 block of B.
    B1 has 8 columns, so 4 blocks of 2x2. Each block needs 7 products. Total: 28.
    
    For the 1x2 * 2x8 part: This is just 1*2*8 = 16 rank-1 terms (each term is A21[j]*B1[j,k]).
    
    For the A12*B2 part: A12 is 2x1, B2 is 1x8. Product is 2x8. Rank = 2*1*8 = 16.
    For the A22*B2 part: 1x1 * 1x8. Rank = 1*1*8 = 8.
    
    Total rank = 28 + 16 + 16 + 8 = 68.
    
    That's an improvement over 72!
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]  # 9x68
    V = [[0] * rank for _ in range(m * p)]  # 24x68
    W = [[0] * rank for _ in range(p * n)]  # 24x68
    
    col = 0
    
    # Part 1: A11 * B1 where A11 is rows 0-1, cols 0-1 of A; B1 is rows 0-1, all 8 cols of B
    # A11: 2x2, B1: 2x8
    # Split B1 into 4 blocks of 2x2: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6], B1[:, 6:8]
    # For each block, use Strassen's 7 products
    
    def strassen_2x2_block(a_row_offset, b_col_offset, start_col):
        """
        Add Strassen decomposition for a 2x2 * 2x2 multiplication.
        a_row_offset: row offset in A (0 or 1 for the 2x2 block)
        b_col_offset: column offset in B's 8 columns (0, 2, 4, 6)
        start_col: starting column in the rank dimension
        Returns the next column index.
        """
        nonlocal col
        # A indices: (a_row_offset + i, j) for i,j in {0,1}
        # B indices: (j, b_col_offset + k) for j in {0,1}, k in {0,1}
        # C indices: (b_col_offset + k)*n + (a_row_offset + i) for the W matrix
        
        # Strassen's 7 products for 2x2*2x2:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10-b11)
        # M4 = a11*(b01-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        
        # C00 = M1 - M4 - M5 - M7
        # C01 = M3 + M5 + M7
        # C10 = M2 + M4 - M6 - M7
        # C11 = M1 + M3 - M2 + M6
        
        products = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1 = (a00+a11)*(b00+b11)
            ([[(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]]),
            # M2 = (a10+a11)*b00
            ([[(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,0,1)]]),
            # M3 = a00*(b10-b11)
            ([[(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,0,-1)]]),
            # M4 = a11*(b01-b00)
            ([[(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,1,-1)]]),
            # M5 = (a00+a01)*b11
            ([[(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,1,1)]]),
            # M6 = (a10-a00)*(b00+b01)
            ([[(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]]),
            # M7 = (a01-a11)*(b10+b11)
            ([[(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]]),
        ]
        
        for prod_idx, (u_coeff, v_coeff, w_coeff) in enumerate(products):
            c = start_col + prod_idx
            for (ri, cj, val) in u_coeff:
                U[(a_row_offset + ri) * m + cj][c] = val
            for (rj, ck, val) in v_coeff:
                V[rj * p + (b_col_offset + ck)][c] = val
            for (ck, ci, val) in w_coeff:
                W[(b_col_offset + ck) * n + (a_row_offset + ci)][c] = val
        
        return start_col + 7
    
    # Apply Strassen to each 2x2 block of B1
    for block in range(4):
        col = strassen_2x2_block(0, block * 2, col)
    
    # Part 2: A12 * B2 where A12 is rows 0-1, col 2 of A (2x1); B2 is row 2, all 8 cols (1x8)
    # C[i,k] += A[i,2] * B[2,k] for i=0,1
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 3: A21 * B1 where A21 is row 2, cols 0-1 of A (1x2); B1 is rows 0-1, all 8 cols (2x8)
    # C[2,k] += sum_j A[2,j] * B[j,k] for j=0,1
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Part 4: A22 * B2 where A22 is A[2,2] (1x1); B2 is row 2, all 8 cols (1x8)
    # C[2,k] += A[2,2] * B[2,k]
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
