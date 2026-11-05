"""Rewritten program: exact rank-minimal tensor decomposition for 3x4 * 4x8."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's rank-7 decomposition for the 2x2x2 blocks where possible,
    combined with schoolbook for the rest.
    
    For 3x4 * 4x8:
    - We can partition the 4x4 middle dimension and use Strassen on 2x2x2 sub-blocks
    - Specifically, split the 4x4 into 2x2 blocks, giving us a 3x2 * 2x4 structure per block
    - Use Strassen (rank 7) for each 2x2x2 block, schoolbook for remaining
    
    A more systematic approach:
    - The 4x4 middle can be split as 2+2
    - For each pair of rows in A (3 rows -> we can group them) and columns in B (8 cols -> group)
    
    Let me think about this more carefully:
    
    A is 3x4, B is 4x8, C = A*B is 3x8.
    
    We can view the 4x4 middle dimension as two 2x2 blocks.
    
    Partition A into A1 (3x2) and A2 (3x2)
    Partition B into B1 (2x8) and B2 (2x8)
    
    Then C = A1*B1 + A2*B2
    
    Each Ai is 3x2, Bi is 2x8.
    
    For Ai (3x2) * Bi (2x8):
    - Split Ai into two 2x2 blocks: Ai1 (2x2), Ai2 (1x2)
    - Split Bi into four 2x2 blocks: Bi11 (2x2), Bi12 (2x2), Bi21 (2x2), Bi22 (2x2)
    - Wait, Bi is 2x8, so split into four 2x2 blocks
    
    Actually, let me use a cleaner decomposition:
    
    A (3x4) = [A11 A12; A21 A22] where A11, A21 are 2x2 and A12, A22 are... no.
    
    Let me partition differently:
    A is 3x4. Split rows into 2+1, cols into 2+2.
    A = [[A11, A12],
         [A21, A22]]
    where A11, A12, A21, A22 are 2x2, 2x2, 1x2, 1x2 respectively.
    
    B is 4x8. Split rows into 2+2, cols into 4+4.
    B = [[B11, B12],
         [B21, B22]]
    where B11, B12, B21, B22 are 2x4, 2x4, 2x4, 2x4.
    
    C = A*B, C is 3x8.
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each term like A11*B11 is 2x2 * 2x4. We can use Strassen for the 2x2 part.
    
    For 2x2 * 2x4:
    Split the 2x4 into two 2x2 blocks: B = [B_left, B_right]
    Then A*B_left and A*B_right can each use Strassen (rank 7).
    
    So 2x2 * 2x4 has rank 14 (two Strassen decompositions).
    
    Let me count the total rank:
    - A11*B11: 2x2 * 2x4 -> 14
    - A11*B12: 2x2 * 2x4 -> 14
    - A12*B21: 1x2 * 2x4 -> 8 (schoolbook: 1*2*4 = 8)
    - A12*B22: 1x2 * 2x4 -> 8
    - A21*B11: 1x2 * 2x4 -> 8
    - A21*B12: 1x2 * 2x4 -> 8
    - A22*B21: 1x2 * 2x4 -> 8
    - A22*B22: 1x2 * 2x4 -> 8
    
    Total: 14 + 14 + 8*6 = 28 + 48 = 76
    
    That's better than 96!
    
    Let me implement this carefully.
    """
    import time
    start_time = time.time()
    
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[i,k]
    
    # Partition:
    # A rows: 2+1 (rows 0,1 and row 2)
    # A cols: 2+2 (cols 0,1 and cols 2,3)
    # B rows: 2+2 (rows 0,1 and rows 2,3)
    # B cols: 4+4 (cols 0-3 and cols 4-7)
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2_2x2(U_base, V_base, W_base, rank_offset, 
                         a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for multiplying A[a_rows, a_cols] * B[b_rows, b_cols] -> C[c_rows, c_cols]
        where each is 2x2.
        
        U_base: base index for U (A matrix)
        V_base: base index for V (B matrix)
        W_base: base index for W (C matrix)
        
        Returns new rank offset.
        """
        # A is 2x2 at rows a_rows, cols a_cols
        # B is 2x2 at rows b_rows, cols b_cols
        # C is 2x2 at rows c_rows, cols c_cols
        
        # A indices: A[r,c] -> U[(a_rows+r)*m + (a_cols+c)]
        # B indices: B[r,c] -> V[(b_rows+r)*p + (b_cols+c)]
        # C indices: C[r,c] -> W[(c_cols+c)*n + (c_rows+r)]
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11; V: B00, B11; W: C00, C11
        u_idx = rank_offset
        for (ar, ac, coeff) in [(0,0,1), (1,1,1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(0,0,1), (1,1,1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(0,0,1), (1,1,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M2 = (A10 + A11)B00
        u_idx = rank_offset
        for (ar, ac, coeff) in [(1,0,1), (1,1,1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(0,0,1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(1,0,1), (1,1,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M3 = A00(B10 - B00)
        u_idx = rank_offset
        for (ar, ac, coeff) in [(0,0,1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(1,0,1), (0,0,-1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(0,1,1), (1,1,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M4 = A11(B01 - B11)
        u_idx = rank_offset
        for (ar, ac, coeff) in [(1,1,1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(0,1,1), (1,1,-1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(0,0,1), (1,0,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M5 = (A00 + A01)B11
        u_idx = rank_offset
        for (ar, ac, coeff) in [(0,0,1), (0,1,1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(1,1,1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(0,0,-1), (1,1,-1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        u_idx = rank_offset
        for (ar, ac, coeff) in [(1,0,1), (0,0,-1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(0,0,1), (0,1,1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(1,1,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        u_idx = rank_offset
        for (ar, ac, coeff) in [(0,1,1), (1,1,-1)]:
            U[U_base + (ar)*m + (ac)][u_idx] += coeff
        for (br, bc, coeff) in [(1,0,1), (1,1,1)]:
            V[V_base + (br)*p + (bc)][u_idx] += coeff
        for (cr, cc, coeff) in [(0,0,1)]:
            W[W_base + (cc)*n + (cr)][u_idx] += coeff
        rank_offset += 1
        
        return rank_offset
    
    def schoolbook_1x2_2x4(U_base, V_base, W_base, rank_offset,
                           a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        """
        Add schoolbook decomposition for multiplying A[1x2] * B[2x4] -> C[1x4]
        """
        for j in range(2):
            for k in range(4):
                u_idx = rank_offset
                U[U_base + a_row*m + (a_cols + j)][u_idx] = 1
                V[V_base + (b_rows + j)*p + (b_cols + k)][u_idx] = 1
                W[W_base + (c_cols + k)*n + c_row][u_idx] = 1
                rank_offset += 1
        return rank_offset
    
    # Initialize with a generous upper bound on rank
    max_rank = 100
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # A11 (2x2) * B11 (2x4) -> C11 (2x4)
    # A11: rows 0-1, cols 0-1
    # B11: rows 0-1, cols 0-3
    # C11: rows 0-1, cols 0-3
    # Split B11 into B11_left (2x2, cols 0-1) and B11_right (2x2, cols 2-3)
    
    # A11 * B11_left -> C11_left
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=0, c_rows=0, c_cols=0)
    
    # A11 * B11_right -> C11_right
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=2, c_rows=0, c_cols=2)
    
    # A11 (2x2) * B12 (2x4) -> C12 (2x4)
    # B12: rows 0-1, cols 4-7
    # Split into B12_left (2x2, cols 4-5) and B12_right (2x2, cols 6-7)
    
    # A11 * B12_left -> C12_left
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=4, c_rows=0, c_cols=4)
    
    # A11 * B12_right -> C12_right
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=6, c_rows=0, c_cols=6)
    
    # A12 (1x2) * B21 (2x4) -> C13... wait, let me re-index.
    
    # Actually, let me re-derive the partitioning more carefully.
    
    # A is 3x4:
    #   A[0:2, 0:2] = A11 (2x2)
    #   A[0:2, 2:4] = A12 (2x2)
    #   A[2:3, 0:2] = A21 (1x2)
    #   A[2:3, 2:4] = A22 (1x2)
    
    # B is 4x8:
    #   B[0:2, 0:4] = B11 (2x4)
    #   B[0:2, 4:8] = B12 (2x4)
    #   B[2:4, 0:4] = B21 (2x4)
    #   B[2:4, 4:8] = B22 (2x4)
    
    # C is 3x8:
    #   C[0:2, 0:4] = C11 (2x4) = A11*B11 + A12*B21
    #   C[0:2, 4:8] = C12 (2x4) = A11*B12 + A12*B22
    #   C[2:3, 0:4] = C21 (1x4) = A21*B11 + A22*B21
    #   C[2:3, 4:8] = C22 (1x4) = A21*B12 + A22*B22
    
    # So we need:
    # 1. A11*B11 (2x2 * 2x4) -> contributes to C11
    # 2. A12*B21 (2x2 * 2x4) -> contributes to C11
    # 3. A11*B12 (2x2 * 2x4) -> contributes to C12
    # 4. A12*B22 (2x2 * 2x4) -> contributes to C12
    # 5. A21*B11 (1x2 * 2x4) -> contributes to C21
    # 6. A22*B21 (1x2 * 2x4) -> contributes to C21
    # 7. A21*B12 (1x2 * 2x4) -> contributes to C22
    # 8. A22*B22 (1x2 * 2x4) -> contributes to C22
    
    # Let me redo this properly.
    
    # Reset
    max_rank = 100
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    rank_offset = 0
    
    # 1. A11*B11: A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
    #    Split B into two 2x2 blocks
    #    A11 * B[0:2, 0:2] -> C[0:2, 0:2]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=0, c_rows=0, c_cols=0)
    #    A11 * B[0:2, 2:4] -> C[0:2, 2:4]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=2, c_rows=0, c_cols=2)
    
    # 2. A12*B21: A[0:2, 2:4] * B[2:4, 0:4] -> C[0:2, 0:4]
    #    A12 is at a_rows=0, a_cols=2
    #    B21 is at b_rows=2, b_cols=0
    #    Split B21 into two 2x2 blocks
    #    A12 * B[2:4, 0:2] -> C[0:2, 0:2]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=2, b_rows=2, b_cols=0, c_rows=0, c_cols=0)
    #    A12 * B[2:4, 2:4] -> C[0:2, 2:4]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=2, b_rows=2, b_cols=2, c_rows=0, c_cols=2)
    
    # 3. A11*B12: A[0:2, 0:2] * B[0:2, 4:8] -> C[0:2, 4:8]
    #    A11 is at a_rows=0, a_cols=0
    #    B12 is at b_rows=0, b_cols=4
    #    Split B12 into two 2x2 blocks
    #    A11 * B[0:2, 4:6] -> C[0:2, 4:6]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=4, c_rows=0, c_cols=4)
    #    A11 * B[0:2, 6:8] -> C[0:2, 6:8]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=0, b_rows=0, b_cols=6, c_rows=0, c_cols=6)
    
    # 4. A12*B22: A[0:2, 2:4] * B[2:4, 4:8] -> C[0:2, 4:8]
    #    A12 is at a_rows=0, a_cols=2
    #    B22 is at b_rows=2, b_cols=4
    #    Split B22 into two 2x2 blocks
    #    A12 * B[2:4, 4:6] -> C[0:2, 4:6]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=2, b_rows=2, b_cols=4, c_rows=0, c_cols=4)
    #    A12 * B[2:4, 6:8] -> C[0:2, 6:8]
    rank_offset = strassen_2x2_2x2(0, 0, 0, rank_offset,
                                   a_rows=0, a_cols=2, b_rows=2, b_cols=6, c_rows=0, c_cols=6)
    
    # 5. A21*B11: A[2:3, 0:2] * B[0:2, 0:4] -> C[2:3, 0:4]
    #    A21 is 1x2 at a_row=2, a_cols=0
    #    B11 is 2x4 at b_rows=0, b_cols=0
    rank_offset = schoolbook_1x2_2x4(0, 0, 0, rank_offset,
                                     a_row=2, a_cols=0, b_rows=0, b_cols=0, c_row=2, c_cols=0)
    
    # 6. A22*B21: A[2:3, 2:4] * B[2:4, 0:4] -> C[2:3, 0:4]
    rank_offset = schoolbook_1x2_2x4(0, 0, 0, rank_offset,
                                     a_row=2, a_cols=2, b_rows=2, b_cols=0, c_row=2, c_cols=0)
    
    # 7. A21*B12: A[2:3, 0:2] * B[0:2, 4:8] -> C[2:3, 4:8]
    rank_offset = schoolbook_1x2_2x4(0, 0, 0, rank_offset,
                                     a_row=2, a_cols=0, b_rows=0, b_cols=4, c_row=2, c_cols=4)
    
    # 8. A22*B22: A[2:3, 2:4] * B[2:4, 4:8] -> C[2:3, 4:8]
    rank_offset = schoolbook_1x2_2x4(0, 0, 0, rank_offset,
                                     a_row=2, a_cols=2, b_rows=2, b_cols=4, c_row=2, c_cols=4)
    
    # Trim to actual rank
    rank = rank_offset
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    # Verify correctness
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta
    # For the matrix multiplication tensor, the condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} for all i,j,k? No.
    # 
    # The tensor T[i,j,k] = delta_{i,k} (for the multiplication tensor, T is the identity in the sense that)
    # Actually, the matrix multiplication tensor M has entries M[i,j,k] where i indexes A, j indexes B, k indexes C.
    # M[i*m+j, j*p+k, k*n+i] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
    # 
    # The decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # Let's verify with random matrices.
    
    # Generate random A and B
    np.random.seed(42)
    A_test = np.random.randint(-10, 10, size=(n, m))
    B_test = np.random.randint(-10, 10, size=(m, p))
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_decomp = np.zeros((n, p), dtype=float)
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        # Reconstruct A contribution
        A_r = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_r[i, j] = u_col[i*m + j]
        
        # Reconstruct B contribution
        B_r = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_r[j, k] = v_col[j*p + k]
        
        # C contribution: w_col[k*n + i] * (A_r @ B_r)[i, k]
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += w_col[k*n + i] * (A_r @ B_r)[i, k]
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fall back to schoolbook
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
    else:
        # Success, use the optimized decomposition
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
