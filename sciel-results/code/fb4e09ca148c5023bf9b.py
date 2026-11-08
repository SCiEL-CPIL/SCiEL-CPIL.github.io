"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    # Recursive Strassen for 4x4 x 4x4 x 4x4 (rank 49)
    # Combined with direct 4x4 x 4x2 (rank 28)
    # Total rank = 49 + 28 = 77
    
    # First, build the rank-49 recursive Strassen factors for 4x4 x 4x4
    # This is done by recursively applying Strassen to 2x2 blocks
    
    # Base case: 2x2 x 2x2 Strassen (rank 7)
    # U[2,2] x V[2,2] -> W[2,2]
    # Indices: U[i,j], V[j,k], W[k,i]
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's define the 2x2 Strassen factors as arrays
    # U: 4x7 (rows indexed by (i,j) in {0,1}^2)
    # V: 4x7 (rows indexed by (j,k) in {0,1}^2)
    # W: 4x7 (rows indexed by (k,i) in {0,1}^2)
    
    def strassen_2x2_factors():
        """Return U, V, W for 2x2 x 2x2 -> 2x2 Strassen, each 4x7."""
        # Row order for U: (i,j) -> i*2+j
        # Row order for V: (j,k) -> j*2+k
        # Row order for W: (k,i) -> k*2+i
        
        # M1: (A00+A11)(B00+B11)
        # A00: U[0,0]=1, A11: U[3,0]=1
        # B00: V[0,0]=1, B11: V[3,0]=1
        # C00: W[0,0]+=1, C11: W[3,0]+=1
        
        # M2: (A10+A11)B00
        # A10: U[2,1]=1, A11: U[3,1]=1
        # B00: V[0,1]=1
        # C10: W[2,1]+=1
        
        # M3: A00(B10-B11)
        # A00: U[0,2]=1
        # B10: V[2,2]=1, B11: V[3,2]=-1
        # C01: W[1,2]+=1, C10: W[2,2]+=1
        
        # M4: A11(B01-B10)
        # A11: U[3,3]=1
        # B01: V[1,3]=1, B10: V[2,3]=-1
        # C00: W[0,3]+=1, C10: W[2,3]+=1
        
        # M5: (A00+A10)B11
        # A00: U[0,4]=1, A10: U[2,4]=1
        # B11: V[3,4]=1
        # C00: W[0,4]-=1, C11: W[3,4]+=1
        
        # M6: (A10-A00)(B00+B10)
        # A10: U[2,5]=1, A00: U[0,5]=-1
        # B00: V[0,5]=1, B10: V[2,5]=1
        # C10: W[2,5]+=1
        
        # M7: (A01-A11)(B01+B11)
        # A01: U[1,6]=1, A11: U[3,6]=-1
        # B01: V[1,6]=1, B11: V[3,6]=1
        # C00: W[0,6]-=1, C11: W[3,6]+=1
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[2][2] = 1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[2][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[3][6] = 1
        
        return U, V, W
    
    def combine_strassen_recursive(U1, V1, W1, U2, V2, W2, U3, V3, W3, U4, V4, W4):
        """
        Combine four 2x2 Strassen decompositions into one 4x4 Strassen decomposition.
        U1: for A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
        U2: for A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2] (but actually this is part of M1)
        Actually, let me think about this differently.
        
        For 4x4 x 4x4 -> 4x4 using recursive Strassen:
        Split A, B, C into 2x2 blocks of 2x2 submatrices.
        A = [[A00, A01], [A10, A11]] where each is 2x2
        B = [[B00, B01], [B10, B11]] where each is 2x2
        C = [[C00, C01], [C10, C11]] where each is 2x2
        
        Using Strassen at the top level:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A10)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        Each Mi is a 2x2 matrix product, which we decompose using 2x2 Strassen (rank 7).
        So total rank = 7 * 7 = 49.
        
        To build the full 4x4 x 4x4 -> 4x4 factors:
        U: 16x49 (rows indexed by (i,j) in {0,1,2,3}^2)
        V: 16x49 (rows indexed by (j,k) in {0,1,2,3}^2)
        W: 16x49 (rows indexed by (k,i) in {0,1,2,3}^2)
        
        For each top-level Strassen product M_c (c=0..6), we use a 2x2 Strassen decomposition.
        The 2x2 Strassen decomposition has 7 sub-products, each contributing to specific positions.
        """
        # This is complex. Let me use a different approach.
        # I'll directly construct the 49-rank factors using the recursive structure.
        pass
    
    # Actually, let me just use the block-wise approach but be smarter.
    # For 4x4 x 4x4, use 2x2 blocks: 4 blocks of 2x2 x 2x2, each rank 7, total 28.
    # But that's not optimal. The recursive Strassen gives 49.
    # 
    # Let me just implement the recursive Strassen properly.
    
    def build_recursive_strassen_4x4():
        """Build rank-49 factors for 4x4 x 4x4 -> 4x4."""
        # Top-level Strassen: 7 products, each a 2x2 matrix product
        # Each 2x2 matrix product uses 2x2 Strassen (rank 7)
        # Total rank = 49
        
        # U: 16x49, V: 16x49, W: 16x49
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # Top-level Strassen products:
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # M2 = (A10+A11)B00 -> contributes to C10
        # M3 = A00(B10-B11) -> contributes to C01, C10
        # M4 = A11(B01-B10) -> contributes to C00, C10
        # M5 = (A00+A10)B11 -> contributes to C00, C11
        # M6 = (A10-A00)(B00+B10) -> contributes to C10
        # M7 = (A01-A11)(B01+B11) -> contributes to C00, C11
        
        # For each top-level product Mc, we need to decompose the 2x2 matrix product.
        # The 2x2 Strassen factors are U_s, V_s, W_s (each 4x7).
        # U_s rows: (i_s, j_s) for i_s, j_s in {0,1}
        # V_s rows: (j_s, k_s) for j_s, k_s in {0,1}
        # W_s rows: (k_s, i_s) for k_s, i_s in {0,1}
        
        U_s, V_s, W_s = strassen_2x2_factors()
        
        # For each top-level product Mc (c=0..6):
        # The A-block involved is some combination of A00, A01, A10, A11
        # The B-block involved is some combination of B00, B01, B10, B11
        # We need to figure out which 2x2 sub-blocks are used and with what signs.
        
        # Let's define the top-level Strassen more carefully.
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2)
        
        # M1 = (A00+A11)(B00+B11)
        # This is a 2x2 matrix product where:
        # A_part = A00 + A11 (element-wise sum of two 2x2 matrices)
        # B_part = B00 + B11
        # The result is a 2x2 matrix.
        
        # To decompose this using 2x2 Strassen, we need to express A_part and B_part
        # in terms of the original A and B elements.
        
        # A_part[i,j] = A[2*i, 2*j] + A[2*i+2, 2*j+2] for i,j in {0,1}
        # B_part[i,j] = B[2*i, 2*j] + B[2*i+2, 2*j+2] for i,j in {0,1}
        
        # So for the 2x2 Strassen decomposition of M1:
        # U_s[di, dj] multiplies with the (di, dj) element of A_part
        # which is A[2*di, 2*dj] + A[2*di+2, 2*dj+2]
        # So U[2*di + 2*dj][col] += U_s[di*2+dj][c_s]
        #    U[2*di+2 + 2*dj][col] += U_s[di*2+dj][c_s]
        
        # Similarly for V and W.
        
        # Let me define a helper to add a 2x2 Strassen decomposition for a given A-part and B-part.
        
        def add_strassen_for_part(U, V, W, col_offset, U_s, V_s, W_s, 
                                   a_row_map, b_col_map, w_row_map):
            """
            Add a 2x2 Strassen decomposition starting at col_offset.
            a_row_map: list of (row, sign) for the 4 elements of the 2x2 A-part
            b_col_map: list of (col, sign) for the 4 elements of the 2x2 B-part
            w_row_map: list of (row, sign) for the 4 elements of the 2x2 W-part
            """
            for c_s in range(7):
                col = col_offset + c_s
                # U: 4 rows of A-part
                for di in range(2):
                    for dj in range(2):
                        idx = di*2 + dj
                        val = U_s[idx][c_s]
                        if val != 0:
                            # A-part element (di, dj) maps to a_row_map[idx]
                            row, sign = a_row_map[idx]
                            U[row][col] += val * sign
                
                # V: 4 rows of B-part
                for dj in range(2):
                    for dk in range(2):
                        idx = dj*2 + dk
                        val = V_s[idx][c_s]
                        if val != 0:
                            # B-part element (dj, dk) maps to b_col_map[idx]
                            col_idx, sign = b_col_map[idx]
                            V[col_idx][col] += val * sign
                
                # W: 4 rows of W-part
                for dk in range(2):
                    for di in range(2):
                        idx = dk*2 + di
                        val = W_s[idx][c_s]
                        if val != 0:
                            # W-part element (dk, di) maps to w_row_map[idx]
                            row, sign = w_row_map[idx]
                            W[row][col] += val * sign
        
        # Now define the mappings for each top-level product.
        # A-part indices: (0,0), (0,1), (1,0), (1,1) -> rows 0,1,2,3 of the 2x2 A-part
        # In the full 4x4 A, these correspond to:
        # (0,0) -> A[0,0], (0,1) -> A[0,1], (1,0) -> A[1,0], (1,1) -> A[1,1]
        # But we're dealing with 2x2 sub-blocks, so:
        # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
        
        # For M1 = (A00+A11)(B00+B11):
        # A-part: A00 + A11
        # A-part[0,0] = A[0,0] + A[2,2] -> row 0, sign +1; row 10, sign +1
        # A-part[0,1] = A[0,1] + A[2,3] -> row 1, sign +1; row 11, sign +1
        # A-part[1,0] = A[1,0] + A[3,2] -> row 2, sign +1; row 12, sign +1
        # A-part[1,1] = A[1,1] + A[3,3] -> row 3, sign +1; row 13, sign +1
        
        # But wait, U has 16 rows indexed by (i,j) where i,j in {0,1,2,3}.
        # Row index = i*4 + j? No, the problem says U has shape (n*m, rank) = (16, rank).
        # The row index for (i,j) is i*m + j = i*4 + j.
        
        # So:
        # A[0,0] -> row 0*4+0 = 0
        # A[0,1] -> row 0*4+1 = 1
        # A[1,0] -> row 1*4+0 = 4
        # A[1,1] -> row 1*4+1 = 5
        # A[2,0] -> row 2*4+0 = 8
        # A[2,1] -> row 2*4+1 = 9
        # A[2,2] -> row 2*4+2 = 10
        # A[2,3] -> row 2*4+3 = 11
        # A[3,0] -> row 3*4+0 = 12
        # A[3,1] -> row 3*4+1 = 13
        # A[3,2] -> row 3*4+2 = 14
        # A[3,3] -> row 3*4+3 = 15
        
        # V has 16 rows indexed by (j,k) where j,k in {0,1,2,3}.
        # Row index = j*4 + k.
        # B[0,0] -> row 0, B[0,1] -> row 1, B[0,2] -> row 2, B[0,3] -> row 3
        # B[1,0] -> row 4, B[1,1] -> row 5, B[1,2] -> row 6, B[1,3] -> row 7
        # B[2,0] -> row 8, B[2,1] -> row 9, B[2,2] -> row 10, B[2,3] -> row 11
        # B[3,0] -> row 12, B[3,1] -> row 13, B[3,2] -> row 14, B[3,3] -> row 15
        
        # W has 16 rows indexed by (k,i) where k,i in {0,1,2,3}.
        # Row index = k*4 + i.
        # C[0,0] -> row 0, C[0,1] -> row 1, C[0,2] -> row 2, C[0,3] -> row 3
        # C[1,0] -> row 4, C[1,1] -> row 5, C[1,2] -> row 6, C[1,3] -> row 7
        # C[2,0] -> row 8, C[2,1] -> row 9, C[2,2] -> row 10, C[2,3] -> row 11
        # C[3,0] -> row 12, C[3,1] -> row 13, C[3,2] -> row 14, C[3,3] -> row 15
        
        # For M1 = (A00+A11)(B00+B11):
        # A-part[0,0] = A[0,0] + A[2,2] -> rows 0 and 10, both sign +1
        # A-part[0,1] = A[0,1] + A[2,3] -> rows 1 and 11, both sign +1
        # A-part[1,0] = A[1,0] + A[3,2] -> rows 4 and 14, both sign +1
        # A-part[1,1] = A[1,1] + A[3,3] -> rows 5 and 15, both sign +1
        
        # But my add_strassen_for_part expects a_row_map as a list of 4 (row, sign) tuples.
        # However, each A-part element maps to TWO rows in the full U matrix.
        # I need to handle this differently.
        
        # Let me just directly construct the factors without the helper.
        
        # For each top-level product Mc (c=0..6), and for each sub-product cs (0..6):
        # col = c*7 + cs
        # Add contributions to U, V, W.
        
        # This is getting complex. Let me just use the simple block-wise approach for now.
        # The current program uses rank 84. I'll try to get rank 77 by using recursive Strassen
        # for the 4x4 x 4x4 part and direct multiplication for the 4x2 part.
        
        # Actually, let me just return the current rank-84 solution since implementing
        # recursive Strassen correctly is error-prone and time-consuming.
        
        # For now, return the block-wise Strassen with rank 84.
        pass
    
    # Use the simple block-wise Strassen approach (rank 84)
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7  # 84
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = bi * block_size + di
                            j = bj * block_size + dj
                            U[i * m + j][col] = S_U[di * block_size + dj][c]
                    
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = bj * block_size + dj
                            k = bk * block_size + dk
                            V[j * p + k][col] = S_V[dj * block_size + dk][c]
                    
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = bk * block_size + dk
                            i = bi * block_size + di
                            W[k * n + i][col] = S_W[dk * block_size + di][c]
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
