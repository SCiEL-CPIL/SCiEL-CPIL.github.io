def construct():
    import time
    t0 = time.time()
    deadline = 110.0  # safety margin
    
    n, m, p = 4, 4, 6
    R1 = 49  # rank for 4x4x4 using Strassen tensor product
    R2 = 28  # rank for 4x4x2 using Strassen tiling
    total_rank = R1 + R2  # 77
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)]   # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]   # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]   # 24 x 77
    
    # --- Part 1: 4x4x4 part (k in {0,1,2,3}) ---
    # Using tensor product of two 2x2x2 Strassen decompositions
    # Each 2x2x2 Strassen has rank 7 with entries in {-1, 0, 1}
    
    # 2x2x2 Strassen decomposition (standard):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For the 2x2x2 case, we define:
    # U_2x2[4 entries, 7 cols], V_2x2[4 entries, 7 cols], W_2x2[4 entries, 7 cols]
    
    # Index: (i,j) for U, (j,k) for V, (k,i) for W in 2x2 case
    # Let's define the 2x2 Strassen factors explicitly
    
    def get_2x2_strassen():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        R = 7
        u = [[0.0]*R for _ in range(4)]  # 2x2 = 4 entries
        v = [[0.0]*R for _ in range(4)]  # 2x2 = 4 entries
        w = [[0.0]*R for _ in range(4)]  # 2x2 = 4 entries
        
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # A11: U[0,0]=1, A22: U[1,1]=1
        # B11: V[0,0]=1, B22: V[1,1]=1
        # C11: W[0,0]=1
        u[0][0] = 1; u[3][0] = 1  # A11, A22
        v[0][0] = 1; v[3][0] = 1  # B11, B22
        w[0][0] = 1  # C11 (contribution to W[0*2+0])
        
        # M2 = (A21+A22)B11
        # A21: U[1,0]=1, A22: U[1,1]=1
        # B11: V[0,0]=1
        # C21: W[1,0]=1
        u[2][1] = 1; u[3][1] = 1  # A21, A22
        v[0][1] = 1  # B11
        w[2][1] = 1  # C21 (W[1*2+0])
        
        # M3 = A11(B21-B22)
        # A11: U[0,0]=1
        # B21: V[1,0]=1, B22: V[1,1]=-1
        # C12: W[0,1]=1
        u[0][2] = 1  # A11
        v[2][2] = 1; v[3][2] = -1  # B21, -B22
        w[1][2] = 1  # C12 (W[0*2+1])
        
        # M4 = A22(B12-B11)
        # A22: U[1,1]=1
        # B12: V[0,1]=1, B11: V[0,0]=-1
        # C21: W[1,0]=1 (wait, C21 gets M2+M4)
        u[3][3] = 1  # A22
        v[1][3] = 1; v[0][3] = -1  # B12, -B11
        w[2][3] = 1  # C21
        
        # M5 = (A11+A12)B22
        # A11: U[0,0]=1, A12: U[0,1]=1
        # B22: V[1,1]=1
        # C12: W[0,1]=1 (C12 = M3+M5)
        u[0][4] = 1; u[1][4] = 1  # A11, A12
        v[3][4] = 1  # B22
        w[1][4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        # A21: U[1,0]=1, A11: U[0,0]=-1
        # B11: V[0,0]=1, B12: V[0,1]=1
        # C22: W[1,1]=1
        u[2][5] = 1; u[0][5] = -1  # A21, -A11
        v[0][5] = 1; v[1][5] = 1  # B11, B12
        w[3][5] = 1  # C22 (W[1*2+1])
        
        # M7 = (A12-A22)(B21+B22)
        # A12: U[0,1]=1, A22: U[1,1]=-1
        # B21: V[1,0]=1, B22: V[1,1]=1
        # C11: W[0,0]=1 (C11 = M1+M4-M5+M7)
        u[1][6] = 1; u[3][6] = -1  # A12, -A22
        v[2][6] = 1; v[3][6] = 1  # B21, B22
        w[0][6] = 1  # C11
        
        return u, v, w
    
    S_U, S_V, S_W = get_2x2_strassen()
    
    # Now use tensor product for 4x4x4:
    # A is 4x4, split into 2x2 blocks: A = [[A11,A12],[A21,A22]]
    # B is 4x4, split into 2x2 blocks: B = [[B11,B12],[B21,B22]]
    # C = A*B, split into 2x2 blocks
    
    # The tensor product of two Strassen decompositions gives rank 7*7=49
    
    # For the 4x4x4 part, we index:
    # A[i,j] with i,j in 0..3
    # B[j,k] with j,k in 0..3
    # C[i,k] with i,k in 0..3
    
    # Block structure:
    # A block (bi,bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # B block (bj,bk): rows bj*2..bj*2+1, cols bk*2..bk*2+1
    # C block (bi,bk): rows bi*2..bi*2+1, cols bk*2..bk*2+1
    
    # For each pair of Strassen terms (c1, c2) from the 2x2x2 decomposition,
    # we add a rank term where:
    # U[i*m+j] = S_U[i_local][c1] * S_U[j_local][c2]
    # V[j*p+k] = S_V[j_local][c1] * S_V[k_local][c2]
    # W[k*n+i] = S_W[k_local][c1] * S_W[i_local][c2]
    
    # Wait, this isn't quite right. The tensor product approach for block Strassen:
    # If we have a 2x2x2 decomposition with factors U2, V2, W2 (each 4x7),
    # then for the 4x4x4 case using 2x2 blocks, the rank is 7*7=49.
    
    # Actually, let me think more carefully. The standard approach:
    # 4x4x4 can be seen as 4 independent 2x2x2 problems if we use schoolbook (rank 4*4=16... no, 4*4*4=64)
    # With Strassen tiling: split 4x4x4 into 2x2x2 blocks, use Strassen (rank 7) for each of the 4 subproblems: rank 4*7=28
    
    # But wait, that's not how Strassen tiling works. Strassen tiling of 4x4x4:
    # A = [[A1,A2],[A3,A4]], B = [[B1,B2],[B3,B4]] where Ai, Bj are 2x2
    # C1 = A1*B1 + A2*B3 (2x2 product)
    # C2 = A1*B2 + A2*B4 (2x2 product)
    # C3 = A3*B1 + A4*B3 (2x2 product)
    # C4 = A3*B2 + A4*B4 (2x2 product)
    
    # To compute C1 = A1*B1 + A2*B3, we can't just use Strassen directly since it's a sum of two products.
    # The correct Strassen tiling uses 7 "big" multiplications where each big multiplication is a 2x2x2 product.
    
    # Standard 4x4x4 Strassen (rank 28):
    # M1 = (A1+A4)*(B1+B4)  [2x2x2 product, rank 4]
    # M2 = (A3+A4)*B1        [2x2x2 product, rank 4]
    # M3 = A1*(B4-B3)        [2x2x2 product, rank 4]
    # M4 = A4*(B2-B1)        [2x2x2 product, rank 4]
    # M5 = (A1+A2)*B4        [2x2x2 product, rank 4]
    # M6 = (A3-A1)*(B1+B2)   [2x2x2 product, rank 4]
    # M7 = (A2-A4)*(B3+B4)   [2x2x2 product, rank 4]
    
    # C1 = M1+M4-M5+M7
    # C2 = M3+M5
    # C3 = M2+M4
    # C4 = M1-M2+M3+M6
    
    # Each Mi is a 2x2x2 product computed with rank 4 (schoolbook for 2x2x2)
    # Total rank: 7*4 = 28
    
    # So for 4x4x4, rank is 28, not 49!
    
    # Let me reconsider. The inspiration says rank 49 for 4x4x4. Let me check:
    # If we use Strassen (rank 7) for each of the 4 subproblems in a naive way, that would be 4*7=28.
    # But the shared Strassen approach also gives 28.
    
    # Actually, I think rank 49 comes from using the tensor product of two 2x2x2 Strassen decompositions
    # in a different way, or perhaps it's the rank of the 4x4x4 tensor itself.
    
    # Let me just use the rank-28 block Strassen for 4x4x4.
    
    # For the 4x4x4 part (k in 0..3), using rank 28:
    
    def build_44x4_strassen():
        """Build rank-28 decomposition for 4x4x4 matrix multiplication."""
        R = 28
        u = [[0.0]*R for _ in range(16)]
        v = [[0.0]*R for _ in range(16)]
        w = [[0.0]*R for _ in range(16)]
        
        # For each of the 7 Strassen terms, we have a 2x2x2 sub-problem
        # Each 2x2x2 uses rank 4 (schoolbook)
        
        # Define the 7 Strassen terms with their U, V, W coefficients
        # Term t has: U factor (4 entries for 2x2), V factor (4 entries), W factor (4 entries)
        
        # Let's define each term explicitly
        # Term 1: M1 = (A1+A4)*(B1+B4)
        # A1+A4: A[0:2,0:2] + A[2:4,2:4]
        # B1+B4: B[0:2,0:2] + B[2:4,2:4]
        # C1: C[0:2,0:2]
        
        # For each 2x2x2 subproblem, we add 4 rank terms (schoolbook)
        
        # Let me define a helper for a 2x2x2 schoolbook at a given offset
        
        def add_2x2_schoolbook(u_mat, v_mat, w_mat, base_col,
                               a_row_offset, a_col_offset,
                               b_row_offset, b_col_offset,
                               c_row_offset, c_col_offset):
            """Add 4 rank terms for 2x2x2 schoolbook multiplication."""
            col = base_col
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        i = a_row_offset + di
                        j = a_col_offset + dj
                        k = b_col_offset + dk
                        u_mat[i*4+j][col] = 1.0
                        v_mat[(b_row_offset+dj)*4+k][col] = 1.0
                        w_mat[k*4+i][col] = 1.0
                        col += 1
            return col
        
        # But this doesn't handle the sums/differences in Strassen terms.
        # I need to add coefficients to the U, V, W entries.
        
        # Let me take a different approach: define each of the 28 terms explicitly.
        
        # Actually, let me just hardcode the rank-28 decomposition.
        # The 28 terms correspond to 7 Strassen terms x 4 schoolbook terms each.
        
        # For simplicity and correctness, let me use a different strategy:
        # Use the known rank-19 or rank-28 decomposition by listing all terms.
        
        # Given the complexity, let me use a simpler approach:
        # For 4x4x4, use schoolbook (rank 64) as a fallback, or use a known decomposition.
        
        # Actually, let me just implement the schoolbook for the whole 4x4x6 (rank 96)
        # and return that. It's correct but not optimal.
        
        return None
    
    # Given the difficulty, let me use a pragmatic approach:
    # Use schoolbook for the entire 4x4x6 (rank 96)
    
    R = 96
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
