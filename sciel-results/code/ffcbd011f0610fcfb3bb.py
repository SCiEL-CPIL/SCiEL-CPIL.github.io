"""Improved program: exploits block structure to reduce rank below schoolbook 30."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # We use a block decomposition approach.
    # A is 2x3, B is 3x5, C is 2x5.
    # Split B's columns into two 2x2 blocks and one 2x1 block.
    # For the 2x2 blocks, use Strassen (rank 7 each).
    # For the 2x1 block, use schoolbook (rank 2*1*1 = 2? No, 2x1 * 1x1 = 2x1, rank 2*1*1=2? 
    #   Actually, for C = A * B_sub where B_sub is 3x1, we need 2*3*1 = 6 multiplications in schoolbook.
    #   But we can do better: C[:, k] = A * B[:, k]. This is a 2x3 * 3x1 product.
    #   Rank of 2x3x1 tensor is 2*3*1 = 6 in schoolbook.
    #   Can we do better? The tensor T_{i,j,k} = delta_{ik} (no, it's sum_j A_{ij} B_{jk}).
    #   For p=1, the tensor is just A * b for a fixed b, which is linear in b.
    #   Actually, for each output column C[:, k], we compute A * B[:, k].
    #   This requires 2*3 = 6 scalar multiplications in schoolbook.
    #   But these are independent for each k, so total rank for the 2x1 part is 6.
    #
    # Wait, let me reconsider. The total rank would be 7 + 7 + 6 = 20? That seems too good.
    # Let me re-examine.
    #
    # Actually, the Strassen decomposition for 2x2 matrix multiplication has rank 7.
    # If we split B into [B1 | B2 | B3] where B1, B2 are 3x2 and B3 is 3x1,
    # then C = [A*B1 | A*B2 | A*B3].
    # A*B1 is 2x2, A*B2 is 2x2, A*B3 is 2x1.
    # 
    # For A*B1 (2x2 = 2x3 * 3x2): this is NOT standard 2x2 matrix multiplication.
    # A is 2x3, B1 is 3x2. The product is 2x2.
    # The tensor for this is T_{i,j,k,l} where i in 0..1, j in 0..1 (output), 
    # and the input is A_{i,m} B_{m,j} for m in 0..2.
    # This is a 2x2x3 tensor (output 2x2, inner dimension 3).
    # The rank of this tensor... in schoolbook it's 2*2*3 = 12? No.
    # 
    # Let me think more carefully. The matrix multiplication tensor M_{n,m,p} has entries
    # T_{i,j,k} = sum_l A_{i,l} B_{l,j} for i in 0..n-1, j in 0..p-1, l in 0..m-1.
    # 
    # For the full problem: n=2, m=3, p=5. Rank in schoolbook = 2*3*5 = 30.
    #
    # The insight from the knowledge: splitting B into 2x2 + 2x2 + 2x1 blocks.
    # But the sub-problems are 2x3x2, 2x3x2, 2x3x1.
    #
    # For 2x3x2: Can we use Strassen? Strassen is for 2x2x2 (n=m=p=2).
    # For 2x3x2, we can split A into two 2x2 and 2x1 blocks? No, A is 2x3.
    # Split A's columns: A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # Then C = A1*B1 + A2*B2 where B1 is 2x2, B2 is 1x2.
    # A1*B1 is 2x2 (Strassen, rank 7), A2*B2 is 2x1 (rank 2*1*1=2? No, 2x1*1x2 = 2x2, rank 2*1*2=4? 
    #   Actually A2 is 2x1, B2 is 1x2, product is 2x2. Rank = 2*1*2 = 4 in schoolbook.
    # So total for 2x3x2 = 7 + 4 = 11? But schoolbook is 2*3*2 = 12. So we save 1.
    #
    # Similarly for 2x3x1: A = [A1 | A2], A1 is 2x2, A2 is 2x1.
    # C = A1*B1 + A2*B2 where B1 is 2x1, B2 is 1x1.
    # A1*B1 is 2x1 (rank 2*2*1 = 4), A2*B2 is 2x1 (rank 2*1*1 = 2).
    # Total = 4 + 2 = 6. Schoolbook is 2*3*1 = 6. No improvement.
    #
    # So total rank = 11 + 11 + 6 = 28. This matches the knowledge!
    #
    # Let me implement this.

    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We'll build U, V, W with rank 28.
    # Block 1: A*B1 where B1 is 3x2 (columns 0,1 of B)
    # Block 2: A*B2 where B2 is 3x2 (columns 2,3 of B)  
    # Block 3: A*B3 where B3 is 3x1 (column 4 of B)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper: add Strassen-like decomposition for 2x2x2 (standard 2x2 mat mult)
    # But we need 2x3x2, which we decompose as (2x2x2) + (2x1x2)
    
    # For block b (b=0,1,2), the sub-problem is 2x3x2 (for b=0,1) or 2x3x1 (for b=2)
    # For b=0,1: split A into A_left (2x2) and A_right (2x1)
    #   A_left * B_left (2x2 * 2x2) -> Strassen rank 7
    #   A_right * B_right (2x1 * 1x2) -> schoolbook rank 4
    # For b=2: split A into A_left (2x2) and A_right (2x1)
    #   A_left * B_left (2x2 * 2x1) -> schoolbook rank 4
    #   A_right * B_right (2x1 * 1x1) -> schoolbook rank 2
    
    # Strassen for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 2x2x2, U indices: A is 2x2, so A_{i,j} maps to U[i*2+j]
    # V indices: B is 2x2, so B_{j,k} maps to V[j*2+k]
    # W indices: C is 2x2, so C_{i,k} maps to W[k*2+i] (trace convention: W[k*n+i])
    
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, a_row_offset, b_col_offset, c_col_offset):
        """Add Strassen decomposition for 2x2x2 sub-problem.
        a_row_offset: row offset in A (0 or 1 for the 2 rows)
        b_col_offset: column offset in B (0, 1, or 2 for the 3 columns, but we use 2)
        c_col_offset: column offset in C (0, 1, or 2 for the 5 columns, but we use 2)
        Returns new column index.
        """
        nonlocal col
        # Strassen multipliers
        # M1 = (A11 + A22) * (B11 + B22)
        U[a_row_offset*3 + 0][col] = 1  # A11
        U[a_row_offset*3 + 1][col] = 1  # A22
        V[b_col_offset*5 + 0][col] = 1  # B11
        V[b_col_offset*5 + 1][col] = 1  # B22
        W[c_col_offset*2 + 0][col] = 1  # C11
        W[c_col_offset*2 + 1][col] = 1  # C22
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[a_row_offset*3 + 1][col] = 1  # A21
        U[a_row_offset*3 + 1][col] = 1  # A22 -- wait, A21 and A22 are different rows
        # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # In our global U, A[i,j] is at U[i*3 + j] for the 2x3 A.
        # For the 2x2 sub-block, A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
        # So A21 is at U[1*3 + 0] = U[3], A22 is at U[1*3 + 1] = U[4]
        # Let me redo with explicit indices.
        
        col -= 1  # undo
        
        # Let me be more careful.
        # A is 2x3. For a 2x2 sub-block using columns c0, c1 of B:
        # A_left uses columns 0,1 of A (i.e., A[0,0], A[0,1], A[1,0], A[1,1])
        # A_right uses column 2 of A (i.e., A[0,2], A[1,2])
        # B_left uses rows 0,1 of B (i.e., B[0,0], B[0,1], B[1,0], B[1,1])
        # B_right uses row 2 of B (i.e., B[2,0], B[2,1])
        
        # Strassen for A_left * B_left:
        # A11 = A[0,0] -> U[0]
        # A12 = A[0,1] -> U[1]
        # A21 = A[1,0] -> U[3]
        # A22 = A[1,1] -> U[4]
        # B11 = B[0,0] -> V[0]
        # B12 = B[0,1] -> V[1]
        # B21 = B[1,0] -> V[5]
        # B22 = B[1,1] -> V[6]
        # C11 = C[0,0] -> W[0]
        # C12 = C[0,1] -> W[1]
        # C21 = C[1,0] -> W[2]
        # C22 = C[1,1] -> W[3]
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[0][col] += 1
        U[4][col] += 1
        V[0][col] += 1
        V[6][col] += 1
        W[0][col] += 1  # C11
        W[3][col] += 1  # C22
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[3][col] += 1
        U[4][col] += 1
        V[0][col] += 1
        W[2][col] += 1  # C21
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[0][col] += 1
        V[1][col] += 1
        V[6][col] -= 1
        W[1][col] += 1  # C12
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[4][col] += 1
        V[5][col] += 1
        V[0][col] -= 1
        W[0][col] += 1  # C11
        W[2][col] += 1  # C21
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[0][col] += 1
        U[1][col] += 1
        V[6][col] += 1
        W[0][col] -= 1  # C11
        W[1][col] += 1  # C12
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[3][col] += 1
        U[0][col] -= 1
        V[0][col] += 1
        V[1][col] += 1
        W[2][col] -= 1  # C21
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[1][col] += 1
        U[4][col] -= 1
        V[5][col] += 1
        V[6][col] += 1
        W[0][col] -= 1  # C11
        W[3][col] += 1  # C22
        col += 1
        
        return col
    
    # For A_right * B_right (2x1 * 1x2 = 2x2):
    # A_right: A[0,2] -> U[2], A[1,2] -> U[5]
    # B_right: B[2,0] -> V[10], B[2,1] -> V[11]
    # C: C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    # Schoolbook: 4 multiplications
    def add_schoolbook_2x1x2(col_start):
        """Add schoolbook for 2x1 * 1x2."""
        nonlocal col
        # C[0,0] = A[0,2] * B[2,0]
        U[2][col] += 1
        V[10][col] += 1
        W[0][col] += 1
        col += 1
        # C[0,1] = A[0,2] * B[2,1]
        U[2][col] += 1
        V[11][col] += 1
        W[1][col] += 1
        col += 1
        # C[1,0] = A[1,2] * B[2,0]
        U[5][col] += 1
        V[10][col] += 1
        W[2][col] += 1
        col += 1
        # C[1,1] = A[1,2] * B[2,1]
        U[5][col] += 1
        V[11][col] += 1
        W[3][col] += 1
        col += 1
        return col
    
    # For A_left * B_left (2x2 * 2x1 = 2x1) for the last block:
    # A_left: A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
    # B_left: B[0,0]->V[0], B[1,0]->V[5]
    # C: C[0,0]->W[0], C[1,0]->W[2]
    # Schoolbook: 4 multiplications
    def add_schoolbook_2x2x1(col_start):
        """Add schoolbook for 2x2 * 2x1."""
        nonlocal col
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        U[0][col] += 1
        V[0][col] += 1
        W[0][col] += 1
        col += 1
        U[1][col] += 1
        V[5][col] += 1
        W[0][col] += 1
        col += 1
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U[3][col] += 1
        V[0][col] += 1
        W[2][col] += 1
        col += 1
        U[4][col] += 1
        V[5][col] += 1
        W[2][col] += 1
        col += 1
        return col
    
    # For A_right * B_right (2x1 * 1x1 = 2x1) for the last block:
    # A_right: A[0,2]->U[2], A[1,2]->U[5]
    # B_right: B[2,0]->V[10]
    # C: C[0,0]->W[0], C[1,0]->W[2]
    # Schoolbook: 2 multiplications
    def add_schoolbook_2x1x1(col_start):
        """Add schoolbook for 2x1 * 1x1."""
        nonlocal col
        U[2][col] += 1
        V[10][col] += 1
        W[0][col] += 1
        col += 1
        U[5][col] += 1
        V[10][col] += 1
        W[2][col] += 1
        col += 1
        return col
    
    # Block 0: columns 0,1 of B
    # C columns 0,1
    add_strassen_2x2x2(0, 0, 0)
    add_schoolbook_2x1x2(col)
    
    # Block 1: columns 2,3 of B
    # C columns 2,3
    # Need to adjust B indices: B[0,2]->V[2], B[0,3]->V[3], B[1,2]->V[7], B[1,3]->V[8]
    # And C indices: C[0,2]->W[4], C[0,3]->W[5], C[1,2]->W[6], C[1,3]->W[7]
    # But my functions assume specific indices. Let me generalize.
    
    # Actually, I need to handle different column offsets. Let me rewrite with parameters.
    
    # Reset and redo with proper parameterization.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 0: B columns 0,1 -> C columns 0,1
    # B_left: B[0,0]->V[0], B[0,1]->V[1], B[1,0]->V[5], B[1,1]->V[6]
    # B_right: B[2,0]->V[10], B[2,1]->V[11]
    # C: C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
    
    # Strassen for 2x2x2 (A_left * B_left)
    # M1
    U[0][col] += 1; U[4][col] += 1; V[0][col] += 1; V[6][col] += 1; W[0][col] += 1; W[3][col] += 1; col += 1
    # M2
    U[3][col] += 1; U[4][col] += 1; V[0][col] += 1; W[2][col] += 1; col += 1
    # M3
    U[0][col] += 1; V[1][col] += 1; V[6][col] -= 1; W[1][col] += 1; col += 1
    # M4
    U[4][col] += 1; V[5][col] += 1; V[0][col] -= 1; W[0][col] += 1; W[2][col] += 1; col += 1
    # M5
    U[0][col] += 1; U[1][col] += 1; V[6][col] += 1; W[0][col] -= 1; W[1][col] += 1; col += 1
    # M6
    U[3][col] += 1; U[0][col] -= 1; V[0][col] += 1; V[1][col] += 1; W[2][col] -= 1; col += 1
    # M7
    U[1][col] += 1; U[4][col] -= 1; V[5][col] += 1; V[6][col] += 1; W[0][col] -= 1; W[3][col] += 1; col += 1
    # Schoolbook 2x1x2 (A_right * B_right)
    U[2][col] += 1; V[10][col] += 1; W[0][col] += 1; col += 1
    U[2][col] += 1; V[11][col] += 1; W[1][col] += 1; col += 1
    U[5][col] += 1; V[10][col] += 1; W[2][col] += 1; col += 1
    U[5][col] += 1; V[11][col] += 1; W[3][col] += 1; col += 1
    
    # Block 1: B columns 2,3 -> C columns 2,3
    # B_left: B[0,2]->V[2], B[0,3]->V[3], B[1,2]->V[7], B[1,3]->V[8]
    # B_right: B[2,2]->V[12], B[2,3]->V[13]
    # C: C[0,2]->W[4], C[0,3]->W[5], C[1,2]->W[6], C[1,3]->W[7]
    
    # Strassen for 2x2x2 (A_left * B_left)
    # M1
    U[0][col] += 1; U[4][col] += 1; V[2][col] += 1; V[8][col] += 1; W[4][col] += 1; W[7][col] += 1; col += 1
    # M2
    U[3][col] += 1; U[4][col] += 1; V[2][col] += 1; W[6][col] += 1; col += 1
    # M3
    U[0][col] += 1; V[3][col] += 1; V[8][col] -= 1; W[5][col] += 1; col += 1
    # M4
    U[4][col] += 1; V[7][col] += 1; V[2][col] -= 1; W[4][col] += 1; W[6][col] += 1; col += 1
    # M5
    U[0][col] += 1; U[1][col] += 1; V[8][col] += 1; W[4][col] -= 1; W[5][col] += 1; col += 1
    # M6
    U[3][col] += 1; U[0][col] -= 1; V[2][col] += 1; V[3][col] += 1; W[6][col] -= 1; col += 1
    # M7
    U[1][col] += 1; U[4][col] -= 1; V[7][col] += 1; V[8][col] += 1; W[4][col] -= 1; W[7][col] += 1; col += 1
    # Schoolbook 2x1x2 (A_right * B_right)
    U[2][col] += 1; V[12][col] += 1; W[4][col] += 1; col += 1
    U[2][col] += 1; V[13][col] += 1; W[5][col] += 1; col += 1
    U[5][col] += 1; V[12][col] += 1; W[6][col] += 1; col += 1
    U[5][col] += 1; V[13][col] += 1; W[7][col] += 1; col += 1
    
    # Block 2: B column 4 -> C column 4
    # B_left: B[0,4]->V[4], B[1,4]->V[9]
    # B_right: B[2,4]->V[14]
    # C: C[0,4]->W[8], C[1,4]->W[9]
    
    # Schoolbook 2x2x1 (A_left * B_left)
    U[0][col] += 1; V[4][col] += 1; W[8][col] += 1; col += 1
    U[1][col] += 1; V[9][col] += 1; W[8][col] += 1; col += 1
    U[3][col] += 1; V[4][col] += 1; W[9][col] += 1; col += 1
    U[4][col] += 1; V[9][col] += 1; W[9][col] += 1; col += 1
    # Schoolbook 2x1x1 (A_right * B_right)
    U[2][col] += 1; V[14][col] += 1; W[8][col] += 1; col += 1
    U[5][col] += 1; V[14][col] += 1; W[9][col] += 1; col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
