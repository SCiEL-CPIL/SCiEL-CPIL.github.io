import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    start_time = time.time()
    time_limit = 110.0

    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Use Strassen's algorithm recursively:
    # 4x4 * 4x9 = (2x2 block matrix) * (2x3 block matrix)
    # 
    # A = [[A11, A12],
    #      [A21, A22]]  where each block is 2x2
    # 
    # B = [[B11, B12, B13],
    #      [B21, B22, B23]]  where each block is 2x3
    #
    # C = A * B has blocks Cij = sum_k Aik * Bkj
    # Each Cij is 2x3
    #
    # Using Strassen for the 2x2 * 2x3 blocks:
    # We can write B as two 2x3 matrices: B_left = [B11, B21]^T and B_right = [B12, B13, B22, B23]
    # Actually, let's think of this differently.
    #
    # The 4x4 * 4x9 product can be seen as computing 4*9 = 36 output entries.
    # Each output entry C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Using Strassen on 2x2 blocks:
    # Split A into 2x2 blocks: A = [[A1, A2], [A3, A4]]
    # Split B into 2x3 blocks: B = [[B1, B2, B3], [B4, B5, B6]]
    #
    # C = [[A1*B1 + A2*B4, A1*B2 + A2*B5, A1*B3 + A2*B6],
    #      [A3*B1 + A4*B4, A3*B2 + A4*B5, A3*B3 + A4*B6]]
    #
    # This is 6 block multiplications (2x2 * 2x3) for each of 6 output positions
    # But we can use Strassen to reduce the number of 2x2*2x2 multiplications
    #
    # Actually, let's use a different approach. The 4x4 * 4x9 can be decomposed
    # by splitting the 9 columns into 3 groups of 3 columns each.
    #
    # For each group of 3 columns, we compute 4x4 * 4x3.
    # 
    # 4x4 * 4x3: Split A into 2x2 blocks, B into 2x3 blocks
    # A = [[A1, A2], [A3, A4]], B = [[B1, B2], [B3, B4]] where B1,B2,B3,B4 are 2x3
    #
    # C = [[A1*B1 + A2*B3, A1*B2 + A2*B4],
    #      [A3*B1 + A4*B3, A3*B2 + A4*B4]]
    #
    # Each entry is a 2x3 matrix. We need to compute 8 products of 2x2 * 2x3.
    # 
    # For 2x2 * 2x3, we can use the fact that this is equivalent to 3 separate 2x2*2x1 products
    # or we can use a Winograd-like approach.
    #
    # Actually, let's use Strassen directly on the 2x2 * 2x2 core and handle the third dimension separately.
    #
    # Better approach: Use the known rank-7 decomposition for 2x2x2 (Strassen)
    # and compose it appropriately.
    #
    # Let me construct the decomposition explicitly.
    #
    # The matrix multiplication tensor for 4x4 * 4x9 has rank at most:
    # - Schoolbook: 4*4*9 = 144
    # - Using Strassen on 2x2 blocks: We need to compute 4x4*4x9
    #   Split into 2x2 blocks: 6 block multiplications of 2x2 * 2x3
    #   Each 2x2 * 2x3 can be done with 7 * 3 = 21 scalar multiplications? No...
    #
    # Let me think more carefully.
    #
    # 4x4 * 4x9:
    # A is 4x4, B is 4x9, C is 4x9
    #
    # Split A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # Split B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks)
    #
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23]]
    #
    # Each Cij is 2x3. There are 6 such blocks.
    # Each block requires 2x2 * 2x3 multiplication.
    #
    # For 2x2 * 2x3:
    # Let X = [[x1, x2], [x3, x4]] (2x2), Y = [[y1, y2, y3], [y4, y5, y6]] (2x3)
    # Z = X * Y = [[x1*y1+x3*y4, x1*y2+x3*y5, x1*y3+x3*y6],
    #              [x2*y1+x4*y4, x2*y2+x4*y5, x2*y3+x4*y6]]
    #
    # This is 12 scalar multiplications in schoolbook.
    # Using Strassen for the 2x2 part: We can compute X * Y1, X * Y2, X * Y3 where Yk is 2x1
    # Each 2x2 * 2x1 can be done with 4 multiplications, so 12 total.
    #
    # Alternatively, we can use a different blocking.
    #
    # Let's try: 4x4 * 4x9 = (4x4 * 4x4) * (4x4 * 4x9 / 4x4) - no, that doesn't work directly.
    #
    # Better: Use the fact that 4x4 * 4x9 can be computed as:
    # 1. Compute 4x4 * 4x4 (using Strassen, rank 49) for the first 4 columns
    # 2. Compute 4x4 * 4x4 (using Strassen, rank 49) for the next 4 columns  
    # 3. Compute 4x4 * 4x1 (using 16 multiplications) for the last column
    # Total: 49 + 49 + 16 = 114
    #
    # But we can do better by using Strassen on the 2x2 blocks more carefully.
    #
    # Actually, let's use the following approach:
    # 4x4 * 4x9: Split B into B1 (4x4) and B2 (4x5)
    # Compute A*B1 using Strassen (rank 49)
    # Compute A*B2 using a decomposition
    #
    # For 4x4 * 4x5: Split B2 into B21 (4x4) and B22 (4x1)
    # A*B21 using Strassen (rank 49)
    # A*B22 using 16 multiplications
    # Total: 49 + 49 + 16 = 114
    #
    # Let's implement the Strassen decomposition for 4x4 * 4x4.
    #
    # Strassen for 4x4:
    # Split A = [[A1, A2], [A3, A4]] (2x2 blocks of 2x2)
    # Split B = [[B1, B2], [B3, B4]] (2x2 blocks of 2x2)
    #
    # M1 = (A1+A4)*(B1+B4)
    # M2 = (A3+A4)*B1
    # M3 = A1*(B2-B4)
    # M4 = A4*(B3-B1)
    # M5 = (A1+A2)*B4
    # M6 = (A3-A1)*(B1+B2)
    # M7 = (A2-A4)*(B3+B4)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix, computed as a 2x2 * 2x2 product.
    # Using Strassen recursively for each 2x2 * 2x2 (rank 7):
    # Total rank = 7 * 7 = 49
    #
    # Now let's construct the explicit decomposition.
    #
    # I'll build the U, V, W matrices for the full 4x4 * 4x9 decomposition.
    #
    # Strategy:
    # 1. Decompose 4x4 * 4x9 into:
    #    - 4x4 * 4x4 for columns 0-3 (rank 49)
    #    - 4x4 * 4x4 for columns 4-7 (rank 49) 
    #    - 4x4 * 4x1 for column 8 (rank 16)
    #    Total: 114
    #
    # Let me implement this.

    # First, let's define the Strassen decomposition for 2x2 * 2x2
    # This gives rank 7
    
    # For 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e+c*g, a*f+c*h], [b*e+d*g, b*f+d*h]]
    #
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    def strassen_2x2_2x2():
        """Return U, V, W for 2x2 * 2x2 decomposition using Strassen (rank 7)."""
        # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        # Indices: U[i*2+j], V[i*2+j], W[i*2+j] for i,j in {0,1}
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # m1 = (a+d)(e+h)
        U[0] = 1; U[3] = 1  # a, d
        V[0] = 1; V[3] = 1  # e, h
        # C11 += m1, C22 += m1
        W[0] = 1; W[3] = 1
        
        # m2 = (c+d)e
        U[2] = 1; U[3] = 1  # c, d
        V[0] = 1  # e
        # C21 += m2
        W[2] = 1
        
        # m3 = a(f-h)
        U[0] = 1  # a
        V[1] = 1; V[3] = -1  # f, -h
        # C12 += m3
        W[1] = 1
        
        # m4 = d(g-e)
        U[3] = 1  # d
        V[2] = 1; V[0] = -1  # g, -e
        # C11 += m4, C21 += m4
        W[0] = 1; W[2] = 1
        
        # m5 = (a+b)h
        U[0] = 1; U[1] = 1  # a, b
        V[3] = 1  # h
        # C11 -= m5, C12 += m5
        W[0] = -1; W[1] = 1
        
        # m6 = (c-a)(e+f)
        U[2] = 1; U[0] = -1  # c, -a
        V[0] = 1; V[1] = 1  # e, f
        # C22 += m6
        W[3] = 1
        
        # m7 = (b-d)(g+h)
        U[1] = 1; U[3] = -1  # b, -d
        V[2] = 1; V[3] = 1  # g, h
        # C11 += m7
        W[0] = 1
        
        return U, V, W

    # Now build 4x4 * 4x4 using Strassen on 2x2 blocks
    def strassen_4x4_4x4():
        """Return U, V, W for 4x4 * 4x4 decomposition using Strassen (rank 49)."""
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        # Strassen formulas for 4x4:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 * 2x2 product, which we decompose with rank 7.
        # So we have 7 * 7 = 49 total rank.
        
        # Block indices: A11 is rows 0-1, cols 0-1; A12 is rows 0-1, cols 2-3
        #                 A21 is rows 2-3, cols 0-1; A22 is rows 2-3, cols 2-3
        # B11 is rows 0-1, cols 0-1; B12 is rows 0-1, cols 2-3
        # B21 is rows 2-3, cols 0-1; B22 is rows 2-3, cols 2-3
        
        # C11 is rows 0-1, cols 0-1; C12 is rows 0-1, cols 2-3
        # C21 is rows 2-3, cols 0-1; C22 is rows 2-3, cols 2-3
        
        # For each of the 7 Strassen multiplications, we use the 2x2*2x2 Strassen.
        # The 7 outer multiplications correspond to the 7 Strassen terms.
        # Each outer term has 7 inner terms (from the 2x2*2x2 Strassen).
        
        # Let's define the outer Strassen structure:
        # For each outer term t (0-6), we have a 2x2*2x2 multiplication.
        # The U, V, W for the outer term tell us which 2x2 blocks to combine.
        # Then for each, we apply the inner 2x2*2x2 Strassen.
        
        # Outer Strassen U, V, W (for 2x2 block level):
        # These are 4x7 matrices where each row corresponds to a 2x2 block.
        
        # Let me define the outer coefficients:
        # M1: U blocks: A11(+1), A22(+1); V blocks: B11(+1), B22(+1)
        # M2: U blocks: A21(+1), A22(+1); V blocks: B11(+1)
        # M3: U blocks: A11(+1); V blocks: B12(+1), B22(-1)
        # M4: U blocks: A22(+1); V blocks: B21(+1), B11(-1)
        # M5: U blocks: A11(+1), A12(+1); V blocks: B22(+1)
        # M6: U blocks: A21(+1), A11(-1); V blocks: B11(+1), B12(+1)
        # M7: U blocks: A12(+1), A22(-1); V blocks: B21(+1), B22(+1)
        
        # Inner Strassen (from strassen_2x2_2x2):
        U_inner, V_inner, W_inner = strassen_2x2_2x2()
        
        # Now compose:
        # For outer term t, inner term s:
        # The U contribution comes from the outer U block coefficients and inner U coefficients.
        
        # Let's define the outer block structure:
        # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
        # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # C blocks: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        
        # Outer coefficients (for the 7 Strassen terms):
        # Each term has U_coeffs[4] and V_coeffs[4] for the 4 blocks.
        outer_U = [
            [1, 0, 0, 1],  # M1: A11, A22
            [0, 0, 1, 1],  # M2: A21, A22
            [1, 0, 0, 0],  # M3: A11
            [0, 0, 0, 1],  # M4: A22
            [1, 1, 0, 0],  # M5: A11, A12
            [-1, 0, 1, 0], # M6: -A11, A21
            [0, 1, 0, -1], # M7: A12, -A22
        ]
        outer_V = [
            [1, 0, 0, 1],  # M1: B11, B22
            [1, 0, 0, 0],  # M2: B11
            [0, 1, 0, -1], # M3: B12, -B22
            [-1, 0, 1, 0], # M4: -B11, B21
            [0, 0, 0, 1],  # M5: B22
            [1, 1, 0, 0],  # M6: B11, B12
            [0, 0, 1, 1],  # M7: B21, B22
        ]
        outer_W = [
            [1, 0, 0, 1],  # M1: C11, C22
            [0, 0, 1, 0],  # M2: C21
            [0, 1, 0, 0],  # M3: C12
            [1, 0, 1, 0],  # M4: C11, C21
            [-1, 1, 0, 0], # M5: -C11, C12
            [0, 0, 0, 1],  # M6: C22
            [1, 0, 0, 0],  # M7: C11
        ]
        
        # Now compose:
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                
                # U: For each row in the 2x2 block (4 rows per block),
                # the coefficient is outer_U[t][block] * U_inner[s][row_in_block]
                for block in range(4):
                    if outer_U[t][block] != 0:
                        # Block 0: rows 0-1, Block 1: rows 2-3, Block 2: rows 4-5, Block 3: rows 6-7
                        # Wait, 4x4 has 16 rows. Each 2x2 block has 4 rows.
                        # A11: rows 0-1, A12: rows 0-1 but cols 2-3... 
                        # Actually, U is indexed by (i,j) where i is row, j is col of A.
                        # A is 4x4, so U has 16 rows: U[i*4+j] for i,j in 0-3.
                        # A11: i in {0,1}, j in {0,1} -> U[0*4+0], U[0*4+1], U[1*4+0], U[1*4+1]
                        # A12: i in {0,1}, j in {2,3} -> U[0*4+2], U[0*4+3], U[1*4+2], U[1*4+3]
                        # A21: i in {2,3}, j in {0,1} -> U[2*4+0], U[2*4+1], U[3*4+0], U[3*4+1]
                        # A22: i in {2,3}, j in {2,3} -> U[2*4+2], U[2*4+3], U[3*4+2], U[3*4+3]
                        
                        # For block b, the rows are:
                        if block == 0:  # A11
                            rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
                        elif block == 1:  # A12
                            rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
                        elif block == 2:  # A21
                            rows = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
                        else:  # A22
                            rows = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
                        
                        for r_idx, r in enumerate(rows):
                            U[r, col] += outer_U[t][block] * U_inner[r_idx, s]
                
                # V: Similarly for B
                for block in range(4):
                    if outer_V[t][block] != 0:
                        if block == 0:  # B11
                            rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
                        elif block == 1:  # B12
                            rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
                        elif block == 2:  # B21
                            rows = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
                        else:  # B22
                            rows = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
                        
                        for r_idx, r in enumerate(rows):
                            V[r, col] += outer_V[t][block] * V_inner[r_idx, s]
                
                # W: Similarly for C
                for block in range(4):
                    if outer_W[t][block] != 0:
                        if block == 0:  # C11
                            rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
                        elif block == 1:  # C12
                            rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
                        elif block == 2:  # C21
                            rows = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
                        else:  # C22
                            rows = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
                        
                        for r_idx, r in enumerate(rows):
                            W[r, col] += outer_W[t][block] * W_inner[r_idx, s]
        
        return U, V, W

    # Now build 4x4 * 4x9:
    # Columns 0-3: use Strassen 4x4*4x4 (rank 49)
    # Columns 4-7: use Strassen 4x4*4x4 (rank 49)
    # Column 8: use schoolbook 4x4*4x1 (rank 16)
    # Total: 114
    
    U_44, V_44, W_44 = strassen_4x4_4x4()
    
    # For columns 0-3, the V matrix needs to be adjusted because B is 4x9, not 4x4.
    # The first 4 columns of B correspond to B11 and B21 in the Strassen decomposition.
    # V_44 has 16 rows for a 4x4 B. For 4x9, V has 36 rows.
    # The first 16 rows of V (for columns 0-3) should match V_44.
    
    # For columns 4-7, we need another Strassen decomposition for B columns 4-7.
    # This is a 4x4 * 4x4 where the B is the next 4 columns.
    
    # For column 8, we use schoolbook: 16 multiplications.
    
    # Let's build the full U, V, W:
    rank_total = 49 + 49 + 16
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((36, rank_total), dtype=float)
    W_full = np.zeros((36, rank_total), dtype=float)
    
    # Part 1: Columns 0-3 (Strassen)
    U_full[:, 0:49] = U_44
    V_full[0:16, 0:49] = V_44  # First 16 rows of V (B columns 0-3)
    # W for C columns 0-3: W_44 has 16 rows for C (4x4). But C is 4x9, so W has 36 rows.
    # C columns 0-3 correspond to rows 0-15 of W (if W is indexed as k*n+i, k in 0-8, i in 0-3)
    # W[k*4+i] for k in 0-8, i in 0-3. So C column 0 is rows 0,4,8,...,32
    # Wait, the trace convention is W[k*n+i, r] where n=4.
    # So W has 36 rows: k in 0-8, i in 0-3. W[k*4+i].
    # For C columns 0-3, we need W for k in 0-8, but C is 4x9 so k in 0-8.
    # Actually, C[i,k] where i in 0-3, k in 0-8.
    # W is indexed as W[k*n+i] = W[k*4+i].
    # For the first 4 columns of C (k=0,1,2,3), we need W[k*4+i] for k in 0-3, i in 0-3.
    # That's rows 0-15 of W.
    W_full[0:16, 0:49] = W_44
    
    # Part 2: Columns 4-7 (Strassen)
    # The B for columns 4-7 is a 4x4 matrix (rows 0-3, cols 4-7).
    # V for this part: rows 16-31 of V_full (since B[j*9+k], j in 0-3, k in 4-7)
    # B[j*9+k] for j in 0-3, k in 4-7. So row index = j*9+k.
    # j=0: k=4,5,6,7 -> rows 4,5,6,7
    # j=1: k=4,5,6,7 -> rows 13,14,15,16
    # j=2: k=4,5,6,7 -> rows 22,23,24,25
    # j=3: k=4,5,6,7 -> rows 31,32,33,34
    # So the rows for B columns 4-7 are: 4,5,6,7,13,14,15,16,22,23,24,25,31,32,33,34
    
    U_full[:, 49:98] = U_44
    # V rows for B columns 4-7:
    v_rows_4_7 = [4,5,6,7,13,14,15,16,22,23,24,25,31,32,33,34]
    for r_idx, r in enumerate(v_rows_4_7):
        V_full[r, 49:98] = V_44[r_idx]
    
    # W for C columns 4-7: W[k*4+i] for k in 4-7, i in 0-3.
    # k=4: i=0,1,2,3 -> rows 16,17,18,19
    # k=5: i=0,1,2,3 -> rows 20,21,22,23
    # k=6: i=0,1,2,3 -> rows 24,25,26,27
    # k=7: i=0,1,2,3 -> rows 28,29,30,31
    w_rows_4_7 = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    for r_idx, r in enumerate(w_rows_4_7):
        W_full[r, 49:98] = W_44[r_idx]
    
    # Part 3: Column 8 (schoolbook)
    # 4x4 * 4x1: 16 multiplications.
    # C[i,8] = sum_j A[i,j] * B[j,8]
    # U[i*4+j] = 1, V[j*9+8] = 1, W[8*4+i] = 1
    for i in range(4):
        for j in range(4):
            col = 98 + i*4 + j
            U_full[i*4+j, col] = 1
            V_full[j*9+8, col] = 1
            W_full[8*4+i, col] = 1
    
    # Verify the decomposition
    T_recon = np.einsum('uc,vc,wc->uvw', U_full, V_full, W_full, optimize=True)
    error = np.linalg.norm(T_recon - T)
    
    if error < 1e-8:
        # Round to integers
        U_rounded = np.round(U_full)
        V_rounded = np.round(V_full)
        W_rounded = np.round(W_full)
        T_recon_r = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded, optimize=True)
        error_r = np.linalg.norm(T_recon_r - T)
        
        if error_r < 1e-9:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
    
    # If the Strassen decomposition didn't work, fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
