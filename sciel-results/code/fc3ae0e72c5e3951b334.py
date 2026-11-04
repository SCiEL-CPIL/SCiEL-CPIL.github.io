"""Matrix multiplication 2x4 * 4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 * 2x2 sub-algorithm
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 2x4 * 4x8:
    # Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B (4x8) into B_T (2x8, rows 0-1) and B_B (2x8, rows 2-3)
    # Result = A_L * B_T + A_R * B_B
    # Each term is (2x2)*(2x8), split into four (2x2)*(2x2) blocks
    # Each (2x2)*(2x2) uses 7 Strassen products
    # Total: 2 * 4 * 7 = 56 products
    
    # Index conventions:
    # U[i*m + j, c] where i in [0,n), j in [0,m)  -> U row index = i*4 + j
    # V[j*p + k, c] where j in [0,m), k in [0,p)  -> V row index = j*8 + k
    # W[k*n + i, c] where k in [0,p), i in [0,n)  -> W row index = k*2 + i
    
    col = 0
    # A_L * B_T: A_L uses A rows 0,1 cols 0,1; B_T uses B rows 0,1 cols 0-7
    # A_L[i,j] = A[i, j] for j in [0,1]
    # B_T[j,k] = B[j, k] for j in [0,1], k in [0,8]
    # Split B_T into 4 blocks of 2x2: B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:6], B_T[:, 6:8]
    
    def strassen_2x2_x_2x2(a_idx, b_block_start, U, V, W, col):
        """
        a_idx: list of 4 (i,j) pairs for A entries, in order [A00, A01, A10, A11]
               where A[i][j] corresponds to U row i*m + j
        b_block_start: column start for the 2x2 block of B
               B[j][k] corresponds to V row j*p + (b_block_start + k_local)
               where k_local in [0,2)
        W entries: C[i][k] corresponds to W row k*n + i
        """
        # A00 = a_idx[0], A01 = a_idx[1], A10 = a_idx[2], A11 = a_idx[3]
        # B00 = B[0][b_block_start], B01 = B[0][b_block_start+1]
        # B10 = B[1][b_block_start], B11 = B[1][b_block_start+1]
        
        # Each Strassen product is a column
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 + A11, V: B00 + B11, W: contributes to C00, C11
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        c = col
        # M1
        U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
        U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
        V[a_idx[0][0]*p + b_block_start][c] = 1  # B00, but wait, a_idx[0] is for A, not B
        # Actually, a_idx gives (i,j) for A. For B, we need the row in B (which is the col in A's right side)
        # Let me reconsider. 
        # For A_L * B_T: A_L[i][j] where i in {0,1}, j in {0,1}
        # B_T[j][k] where j in {0,1}, k in {0,8}
        # So for the 2x2 block B_T[:, c0:c0+2], B_T[j][k] = B[j][c0 + k_local]
        # V row for B_T[j][k] is j*p + (c0 + k_local) = j*8 + c0 + k_local
        # But a_idx[0] is (i,j) for A. The j in a_idx is the column of A, which is the row of B_T.
        # So B_T row = a_idx[0][1]? No, that's the column of A.
        # Wait, A_L is 2x2: rows 0,1 of A, cols 0,1 of A.
        # B_T is 2x8: rows 0,1 of B, cols 0-7 of B.
        # The multiplication A_L * B_T: result[i][k] = sum_j A_L[i][j] * B_T[j][k]
        # So A_L[i][j] = A[i][j] (i in {0,1}, j in {0,1})
        # B_T[j][k] = B[j][k] (j in {0,1}, k in {0,8})
        # For the 2x2 block at columns c0, c0+1 of B_T:
        # B_T[j][c0] = B[j][c0], B_T[j][c0+1] = B[j][c0+1]
        # V row for B[j][k] is j*p + k = j*8 + k
        # So V row for B_T[j][c0] is j*8 + c0, for B_T[j][c0+1] is j*8 + c0+1
        
        # a_idx[0] = (i,j) for A00, so A00 = A[i][j], U row = i*m + j = i*4 + j
        # For M1 = (A00 + A11)(B00 + B11):
        # U: A00 + A11 -> U[i0*4+j0][c] += 1, U[i1*4+j1][c] += 1 where (i0,j0)=a_idx[0], (i1,j1)=a_idx[3]
        # V: B00 + B11 -> V[j0*8+c0][c] += 1, V[j1*8+c0+1][c] += 1 where j0=a_idx[0][1]... 
        # No! j in a_idx is the column index of A, which equals the row index of B_T.
        # So B_T row for A's column j is j. So B00 = B_T[0][c0] = B[0][c0], V row = 0*8 + c0
        # B11 = B_T[1][c0+1] = B[1][c0+1], V row = 1*8 + c0+1
        
        # Let me redefine: a_idx entries are (i, j) where i is A row, j is A col.
        # For B, the row is j (A's col), and the col is c0 + k_local.
        
        U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
        U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
        V[a_idx[0][1]*p + b_block_start][c] = 1
        V[a_idx[3][1]*p + b_block_start + 1][c] = 1
        # W: C00 += M1, C11 += M1
        # C00: W row = 0*n + 0 = 0, C11: W row = 1*n + 1 = 3
        W[0*n + 0][c] = 1
        W[1*n + 1][c] = 1
        c += 1
        
        # M2 = (A10 + A11)B00
        U[a_idx[2][0]*m + a_idx[2][1]][c] = 1
        U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
        V[a_idx[2][1]*p + b_block_start][c] = 1
        W[1*n + 0][c] = 1  # C10 += M2
        W[1*n + 1][c] = -1 # C11 -= M2
        c += 1
        
        # M3 = A00(B10 - B11)
        U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
        V[a_idx[3][1]*p + b_block_start][c] = 1   # B10
        V[a_idx[3][1]*p + b_block_start + 1][c] = -1 # B11
        W[0*n + 1][c] = 1  # C01 += M3
        W[1*n + 1][c] = 1  # C11 += M3
        c += 1
        
        # M4 = A11(B01 - B00)
        U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
        V[a_idx[0][1]*p + b_block_start + 1][c] = 1  # B01
        V[a_idx[0][1]*p + b_block_start][c] = -1     # B00
        W[0*n + 0][c] = 1  # C00 += M4
        W[1*n + 0][c] = 1  # C10 += M4
        c += 1
        
        # M5 = (A00 + A01)B11
        U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
        U[a_idx[1][0]*m + a_idx[1][1]][c] = 1
        V[a_idx[3][1]*p + b_block_start + 1][c] = 1  # B11
        W[0*n + 0][c] = -1  # C00 -= M5
        W[0*n + 1][c] = 1   # C01 += M5
        c += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a_idx[2][0]*m + a_idx[2][1]][c] = 1
        U[a_idx[0][0]*m + a_idx[0][1]][c] = -1
        V[a_idx[0][1]*p + b_block_start][c] = 1    # B00
        V[a_idx[0][1]*p + b_block_start + 1][c] = 1 # B01
        W[0*n + 0][c] = -1  # C00 -= M6
        c += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_idx[1][0]*m + a_idx[1][1]][c] = 1
        U[a_idx[3][0]*m + a_idx[3][1]][c] = -1
        V[a_idx[3][1]*p + b_block_start][c] = 1     # B10
        V[a_idx[3][1]*p + b_block_start + 1][c] = 1 # B11
        W[0*n + 0][c] = -1  # C00 -= M7
        W[1*n + 1][c] = 1   # C11 += M7
        c += 1
        
        return c
    
    # A_L * B_T
    # A_L[i][j] = A[i][j], i in {0,1}, j in {0,1}
    # a_idx for A_L: [(0,0), (0,1), (1,0), (1,1)]
    a_idx_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # B_T blocks: cols 0-1, 2-3, 4-5, 6-7
    for blk in range(4):
        col = strassen_2x2_x_2x2(a_idx_L, blk * 2, U, V, W, col)
    
    # A_R * B_B
    # A_R[i][j] = A[i][j+2], i in {0,1}, j in {0,1}
    # a_idx for A_R: [(0,2), (0,3), (1,2), (1,3)]
    a_idx_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
    # B_B is rows 2,3 of B, cols 0-7
    # For B_B, the row in B is 2 + j_local where j_local in {0,1}
    # So V row for B_B[j_local][k] is (2+j_local)*8 + k
    # In the strassen function, V row uses a_idx[j][1]*p + b_block_start
    # But a_idx[j][1] is the column of A, which for A_R is 2,3. That's not the row of B_B.
    # I need to adjust: for B_B, the row offset is 2.
    # Let me handle this separately or add an offset parameter.
    
    # For A_R * B_B: A_R[i][j] = A[i][j+2], B_B[j][k] = B[j+2][k]
    # So in the Strassen products, B entries use row (2 + j_local) in B, not j from A.
    # I'll write a modified version.
    
    col = strassen_2x2_x_2x2_offset(a_idx_R, 2, 0, U, V, W, col)
    for blk in range(1, 4):
        col = strassen_2x2_x_2x2_offset(a_idx_R, 2, blk * 2, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}

def strassen_2x2_x_2x2_offset(a_idx, b_row_offset, b_block_start, U, V, W, col):
    """
    Same as strassen_2x2_x_2x2 but B rows are offset by b_row_offset.
    B[j][k] where j is in {b_row_offset, b_row_offset+1}, k in {b_block_start, b_block_start+1}
    V row for B[j][k] is j*p + k
    """
    n, m, p = 2, 4, 8
    c = col
    # a_idx: [(i0,j0), (i1,j1), (i2,j2), (i3,j3)] for A00, A01, A10, A11
    # B00 = B[b_row_offset][b_block_start], B01 = B[b_row_offset][b_block_start+1]
    # B10 = B[b_row_offset+1][b_block_start], B11 = B[b_row_offset+1][b_block_start+1]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
    U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
    V[b_row_offset*p + b_block_start][c] = 1
    V[(b_row_offset+1)*p + b_block_start + 1][c] = 1
    W[0*n + 0][c] = 1
    W[1*n + 1][c] = 1
    c += 1
    
    # M2 = (A10 + A11)B00
    U[a_idx[2][0]*m + a_idx[2][1]][c] = 1
    U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
    V[b_row_offset*p + b_block_start][c] = 1
    W[1*n + 0][c] = 1
    W[1*n + 1][c] = -1
    c += 1
    
    # M3 = A00(B10 - B11)
    U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
    V[(b_row_offset+1)*p + b_block_start][c] = 1
    V[(b_row_offset+1)*p + b_block_start + 1][c] = -1
    W[0*n + 1][c] = 1
    W[1*n + 1][c] = 1
    c += 1
    
    # M4 = A11(B01 - B00)
    U[a_idx[3][0]*m + a_idx[3][1]][c] = 1
    V[b_row_offset*p + b_block_start + 1][c] = 1
    V[b_row_offset*p + b_block_start][c] = -1
    W[0*n + 0][c] = 1
    W[1*n + 0][c] = 1
    c += 1
    
    # M5 = (A00 + A01)B11
    U[a_idx[0][0]*m + a_idx[0][1]][c] = 1
    U[a_idx[1][0]*m + a_idx[1][1]][c] = 1
    V[(b_row_offset+1)*p + b_block_start + 1][c] = 1
    W[0*n + 0][c] = -1
    W[0*n + 1][c] = 1
    c += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[a_idx[2][0]*m + a_idx[2][1]][c] = 1
    U[a_idx[0][0]*m + a_idx[0][1]][c] = -1
    V[b_row_offset*p + b_block_start][c] = 1
    V[b_row_offset*p + b_block_start + 1][c] = 1
    W[0*n + 0][c] = -1
    c += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[a_idx[1][0]*m + a_idx[1][1]][c] = 1
    U[a_idx[3][0]*m + a_idx[3][1]][c] = -1
    V[(b_row_offset+1)*p + b_block_start][c] = 1
    V[(b_row_offset+1)*p + b_block_start + 1][c] = 1
    W[0*n + 0][c] = -1
    W[1*n + 1][c] = 1
    c += 1
    
    return c
# EVOLVE-BLOCK-END
