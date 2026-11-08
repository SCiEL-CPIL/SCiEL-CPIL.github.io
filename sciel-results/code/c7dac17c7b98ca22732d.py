"""Rewritten program for 3x4x6 matrix multiplication tensor decomposition."""
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x6 matrix.
    
    Strategy: Use a block decomposition approach.
    - Split the 3x4 matrix A into two blocks: A1 (2x4) and A2 (1x4)
    - Multiply each block with B (4x6)
    - For A1*B1 (2x4x6): Use Strassen's algorithm for 2x2x2 blocks
    - For A2*B2 (1x4x6): Use schoolbook (6 dot products of length 4)
    
    Total rank: 42 (Strassen for 2x4x6) + 24 (schoolbook for 1x4x6) = 66
    
    This is better than the schoolbook rank of 72.
    """
    n, m, p = 3, 4, 6
    
    # We'll build the U, V, W matrices as lists of lists
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (18, rank)
    
    # First, let's determine the total rank
    # For 2x4x6 using Strassen: 
    #   2x4 can be seen as two 2x2 blocks horizontally
    #   4x6 can be seen as four 2x2 blocks horizontally (since 6=2+2+2)
    #   Actually, let's think more carefully
    #   
    # Let me reconsider. For multiplying a 2x4 matrix by a 4x6 matrix:
    # A is 2x4, B is 4x6, C is 2x6
    # We can split A into [A11 A12] where A11, A12 are 2x2
    # B into [B11; B12] where B11, B12 are 2x4
    # C = A11*B11 + A12*B12
    # Each 2x2*2x4 multiplication: rank 7*2 = 14 (since 2x4 = two 2x2 columns)
    # Total for 2x4x6: 14 + 14 = 28
    #
    # For 1x4x6:
    # A is 1x4, B is 4x6, C is 1x6
    # This is 6 dot products of length 4
    # Each dot product has rank 4
    # Total: 6*4 = 24
    #
    # Grand total: 28 + 24 = 52
    
    # Wait, let me recalculate. Strassen's 2x2x2 has rank 7.
    # For 2x2 * 2x4: 
    #   A is 2x2, B is 2x4 = [B1 B2] where B1, B2 are 2x2
    #   C = A*B1 + A*B2
    #   Each A*Bk is a 2x2*2x2 multiplication, rank 7
    #   So 2x2*2x4 has rank 14
    # For 2x4 * 4x6:
    #   A is 2x4 = [A1 A2] where A1, A2 are 2x2
    #   B is 4x6 = [B1; B2] where B1, B2 are 2x6
    #   C = A1*B1 + A2*B2
    #   Each Ai*Bi is 2x2*2x6
    #   2x2*2x6: A is 2x2, B is 2x6 = [B1 B2 B3] where each Bk is 2x2
    #   C = A*B1 + A*B2 + A*B3
    #   Each A*Bk is 2x2*2x2, rank 7
    #   So 2x2*2x6 has rank 21
    #   So 2x4*4x6 has rank 21 + 21 = 42
    #
    # For 1x4*4x6:
    #   A is 1x4, B is 4x6, C is 1x6
    #   This is 6 dot products of length 4
    #   Each dot product has rank 4
    #   Total: 24
    #
    # Grand total: 42 + 24 = 66
    
    total_rank = 66
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper function for Strassen 2x2x2
    def strassen_2x2x2(u_offset, v_offset, w_offset, row_offset_u, row_offset_v, row_offset_w):
        """
        Add Strassen's 2x2x2 decomposition.
        
        For matrices:
        A = [[a11, a12], [a21, a22]]
        B = [[b11, b12], [b21, b22]]
        C = A*B
        
        Strassen computes 7 products:
        M1 = (a11+a22)*(b11+b22)
        M2 = (a21+a22)*b11
        M3 = a11*(b12-b22)
        M4 = a22*(b21-b11)
        M5 = (a11+a12)*b22
        M6 = (a21-a11)*(b11+b12)
        M7 = (a12-a22)*(b21+b22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        
        U rows correspond to A elements: index by (i*m + j) where i is row, j is col
        V rows correspond to B elements: index by (j*p + k) where j is row, k is col
        W rows correspond to C elements: index by (k*n + i) where i is row, k is col
        """
        # The 7 Strassen products
        # Each product uses specific U and V entries
        
        # M1 = (a11+a22)*(b11+b22)
        # U: a11 at (row_offset_u + 0), a22 at (row_offset_u + 3)  [for 2x2: row0col0, row0col1, row1col0, row1col1]
        # Wait, I need to be careful with indexing.
        
        # For a 2x2 matrix A stored in U:
        # U[row_offset_u + 0][col] corresponds to a11 (row 0, col 0)
        # U[row_offset_u + 1][col] corresponds to a12 (row 0, col 1)
        # U[row_offset_u + 2][col] corresponds to a21 (row 1, col 0)
        # U[row_offset_u + 3][col] corresponds to a22 (row 1, col 1)
        
        # For a 2x2 matrix B stored in V:
        # V[row_offset_v + 0][col] corresponds to b11 (row 0, col 0)
        # V[row_offset_v + 1][col] corresponds to b12 (row 0, col 1)
        # V[row_offset_v + 2][col] corresponds to b21 (row 1, col 0)
        # V[row_offset_v + 3][col] corresponds to b22 (row 1, col 1)
        
        # For the resulting 2x2 matrix C stored in W:
        # W[row_offset_w + 0][col] corresponds to c11 (row 0, col 0)
        # W[row_offset_w + 1][col] corresponds to c12 (row 0, col 1)
        # W[row_offset_w + 2][col] corresponds to c21 (row 1, col 0)
        # W[row_offset_w + 3][col] corresponds to c22 (row 1, col 1)
        
        # M1
        col = u_offset
        U[row_offset_u + 0][col] = 1  # a11
        U[row_offset_u + 3][col] = 1  # a22
        V[row_offset_v + 0][col] = 1  # b11
        V[row_offset_v + 3][col] = 1  # b22
        W[row_offset_w + 0][col] = 1  # c11 += M1
        W[row_offset_w + 3][col] = 1  # c22 += M1
        
        # M2
        col = u_offset + 1
        U[row_offset_u + 2][col] = 1  # a21
        U[row_offset_u + 3][col] = 1  # a22
        V[row_offset_v + 0][col] = 1  # b11
        W[row_offset_w + 2][col] = 1  # c21 += M2
        W[row_offset_w + 3][col] = -1  # c22 -= M2
        
        # M3
        col = u_offset + 2
        U[row_offset_u + 0][col] = 1  # a11
        V[row_offset_v + 1][col] = 1  # b12
        V[row_offset_v + 3][col] = -1  # -b22
        W[row_offset_w + 1][col] = 1  # c12 += M3
        W[row_offset_w + 3][col] = 1  # c22 += M3
        
        # M4
        col = u_offset + 3
        U[row_offset_u + 3][col] = 1  # a22
        V[row_offset_v + 2][col] = 1  # b21
        V[row_offset_v + 0][col] = -1  # -b11
        W[row_offset_w + 0][col] = 1  # c11 += M4
        W[row_offset_w + 2][col] = 1  # c21 += M4
        
        # M5
        col = u_offset + 4
        U[row_offset_u + 0][col] = 1  # a11
        U[row_offset_u + 1][col] = 1  # a12
        V[row_offset_v + 3][col] = 1  # b22
        W[row_offset_w + 0][col] = -1  # c11 -= M5
        W[row_offset_w + 1][col] = 1  # c12 += M5
        
        # M6
        col = u_offset + 5
        U[row_offset_u + 2][col] = 1  # a21
        U[row_offset_u + 0][col] = -1  # -a11
        V[row_offset_v + 0][col] = 1  # b11
        V[row_offset_v + 1][col] = 1  # b12
        W[row_offset_w + 3][col] = 1  # c22 += M6
        
        # M7
        col = u_offset + 6
        U[row_offset_u + 1][col] = 1  # a12
        U[row_offset_u + 3][col] = -1  # -a22
        V[row_offset_v + 2][col] = 1  # b21
        V[row_offset_v + 3][col] = 1  # b22
        W[row_offset_w + 0][col] = 1  # c11 += M7
        
        return 7  # number of columns used
    
    # Now let's build the decomposition
    col_ptr = 0
    
    # Part 1: A1 * B where A1 is 2x4 (first 2 rows of A), B is 4x6
    # A1 is 2x4, B is 4x6
    # Split A1 = [A11 A12] where A11, A12 are 2x2
    # Split B = [B1; B2] where B1, B2 are 2x6
    # C1 = A11*B1 + A12*B2
    # Each Aik*Bk is 2x2 * 2x6
    
    # For 2x2 * 2x6:
    # A is 2x2, B is 2x6 = [B1 B2 B3] where each Bj is 2x2
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bj is 2x2*2x2, rank 7
    # Total for 2x2*2x6: 21
    
    def compute_2x2_2x6(u_row_off, v_row_off, w_row_off, u_col_off, v_col_off, w_col_off):
        """
        Compute 2x2 * 2x6 using 3 Strassen 2x2x2 blocks.
        A is 2x2, B is 2x6 = [B1 B2 B3] (each 2x2), C is 2x6 = [C1 C2 C3] (each 2x2)
        """
        # A*B1
        # A rows: u_row_off + [0,1,2,3]
        # B1 rows: v_row_off + [0,1,2,3]
        # C1 rows: w_row_off + [0,1,2,3]
        col_ptr_local = u_col_off
        n1 = strassen_2x2x2(col_ptr_local, col_ptr_local, col_ptr_local, 
                            u_row_off, v_row_off, w_row_off)
        
        # A*B2
        # B2 rows: v_row_off + [4,5,6,7] (since B is 2x6, stored as 2x2 blocks)
        # C2 rows: w_row_off + [6,7,8,9]
        col_ptr_local = u_col_off + 7
        n2 = strassen_2x2x2(col_ptr_local, col_ptr_local, col_ptr_local,
                            u_row_off, v_row_off + 4, w_row_off + 6)
        
        # A*B3
        # B3 rows: v_row_off + [8,9,10,11]
        # C3 rows: w_row_off + [12,13,14,15]
        col_ptr_local = u_col_off + 14
        n3 = strassen_2x2x2(col_ptr_local, col_ptr_local, col_ptr_local,
                            u_row_off, v_row_off + 8, w_row_off + 12)
        
        return 21
    
    # A11 is at rows 0-1 of A, cols 0-1 of A
    # In U: rows 0,1,4,5 (since U row index = i*m + j, m=4)
    #   i=0,j=0 -> 0
    #   i=0,j=1 -> 1
    #   i=1,j=0 -> 4
    #   i=1,j=1 -> 5
    # But we need consecutive rows for Strassen. Let me handle this differently.
    
    # Actually, I think I'm overcomplicating this. Let me use a cleaner approach.
    
    # Let me restart with a simpler strategy: just use schoolbook but with some optimization.
    # Or, let me carefully implement the block decomposition.
    
    # Clear the matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_ptr = 0
    
    # Part 1: First 2 rows of A (2x4) times B (4x6)
    # A_sub is 2x4: rows 0-1 of A, all 4 columns
    # In U, A_sub entries are at rows: 0,1,4,5 (i*m+j for i in {0,1}, j in {0,1,2,3})
    # B is 4x6: rows 0-3 of B, all 6 columns
    # In V, B entries are at rows: 0-23 (j*p+k for j in {0,1,2,3}, k in {0,1,2,3,4,5})
    # C_sub is 2x6: rows 0-1 of C, all 6 columns
    # In W, C_sub entries are at rows: 0,3,6,9,12,15 (k*n+i for i in {0,1}, k in {0,1,2,3,4,5})
    #   i=0: k*n+i = k*3+0 = 0,3,6,9,12,15
    #   i=1: k*n+i = k*3+1 = 1,4,7,10,13,16
    
    # Split A_sub into A11 (2x2, cols 0-1) and A12 (2x2, cols 2-3)
    # Split B into B_top (2x6, rows 0-1) and B_bottom (2x6, rows 2-3)
    # C_sub = A11*B_top + A12*B_bottom
    
    # A11*B_top: 2x2 * 2x6
    # A11 in U: rows 0,1,4,5
    # B_top in V: rows 0,1,2,3 (j=0,1; k=0..5) -> actually 12 rows: 0-5 for j=0, 6-11 for j=1
    # Wait, V row index = j*p + k, p=6
    # j=0: 0,1,2,3,4,5
    # j=1: 6,7,8,9,10,11
    # So B_top is V rows 0-11
    
    # Let me define helper functions more carefully
    
    def add_strassen_2x2(u_rows, v_rows, w_rows, col_start):
        """
        Add a Strassen 2x2x2 decomposition.
        u_rows: list of 4 U row indices for the 2x2 A matrix (in order a11,a12,a21,a22)
        v_rows: list of 4 V row indices for the 2x2 B matrix (in order b11,b12,b21,b22)
        w_rows: list of 4 W row indices for the 2x2 C matrix (in order c11,c12,c21,c22)
        col_start: starting column index
        Returns: number of columns used (7)
        """
        c = col_start
        
        # M1 = (a11+a22)*(b11+b22)
        U[u_rows[0]][c] = 1
        U[u_rows[3]][c] = 1
        V[v_rows[0]][c] = 1
        V[v_rows[3]][c] = 1
        W[w_rows[0]][c] = 1
        W[w_rows[3]][c] = 1
        c += 1
        
        # M2 = (a21+a22)*b11
        U[u_rows[2]][c] = 1
        U[u_rows[3]][c] = 1
        V[v_rows[0]][c] = 1
        W[w_rows[2]][c] = 1
        W[w_rows[3]][c] = -1
        c += 1
        
        # M3 = a11*(b12-b22)
        U[u_rows[0]][c] = 1
        V[v_rows[1]][c] = 1
        V[v_rows[3]][c] = -1
        W[w_rows[1]][c] = 1
        W[w_rows[3]][c] = 1
        c += 1
        
        # M4 = a22*(b21-b11)
        U[u_rows[3]][c] = 1
        V[v_rows[2]][c] = 1
        V[v_rows[0]][c] = -1
        W[w_rows[0]][c] = 1
        W[w_rows[2]][c] = 1
        c += 1
        
        # M5 = (a11+a12)*b22
        U[u_rows[0]][c] = 1
        U[u_rows[1]][c] = 1
        V[v_rows[3]][c] = 1
        W[w_rows[0]][c] = -1
        W[w_rows[1]][c] = 1
        c += 1
        
        # M6 = (a21-a11)*(b11+b12)
        U[u_rows[2]][c] = 1
        U[u_rows[0]][c] = -1
        V[v_rows[0]][c] = 1
        V[v_rows[1]][c] = 1
        W[w_rows[3]][c] = 1
        c += 1
        
        # M7 = (a12-a22)*(b21+b22)
        U[u_rows[1]][c] = 1
        U[u_rows[3]][c] = -1
        V[v_rows[2]][c] = 1
        V[v_rows[3]][c] = 1
        W[w_rows[0]][c] = 1
        
        return 7
    
    # A11 rows in U: i=0,1; j=0,1 -> [0, 1, 4, 5]
    a11_u_rows = [0, 1, 4, 5]
    
    # A12 rows in U: i=0,1; j=2,3 -> [2, 3, 6, 7]
    a12_u_rows = [2, 3, 6, 7]
    
    # B_top rows in V: j=0,1; k=0..5 -> 12 rows
    # Split into 3 blocks of 2x2:
    # B_top1: j=0,1; k=0,1 -> V rows [0, 6, 1, 7] (b11,b12,b21,b22)
    # Wait, V row index = j*p + k = j*6 + k
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 6
    # j=1,k=1 -> 7
    # So B_top1 v_rows = [0, 1, 6, 7]
    b_top1_v_rows = [0, 1, 6, 7]
    
    # B_top2: j=0,1; k=2,3 -> V rows [2, 3, 8, 9]
    b_top2_v_rows = [2, 3, 8, 9]
    
    # B_top3: j=0,1; k=4,5 -> V rows [4, 5, 10, 11]
    b_top3_v_rows = [4, 5, 10, 11]
    
    # C_sub1 (2x2) rows in W: i=0,1; k=0,1
    # W row index = k*n + i = k*3 + i
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 3
    # k=1,i=1 -> 4
    c_sub1_w_rows = [0, 1, 3, 4]
    
    # C_sub2 (2x2) rows in W: i=0,1; k=2,3
    # k=2,i=0 -> 6
    # k=2,i=1 -> 7
    # k=3,i=0 -> 9
    # k=3,i=1 -> 10
    c_sub2_w_rows = [6, 7, 9, 10]
    
    # C_sub3 (2x2) rows in W: i=0,1; k=4,5
    # k=4,i=0 -> 12
    # k=4,i=1 -> 13
    # k=5,i=0 -> 15
    # k=5,i=1 -> 16
    c_sub3_w_rows = [12, 13, 15, 16]
    
    # A11 * B_top1
    col_ptr += add_strassen_2x2(a11_u_rows, b_top1_v_rows, c_sub1_w_rows, col_ptr)
    # A11 * B_top2
    col_ptr += add_strassen_2x2(a11_u_rows, b_top2_v_rows, c_sub2_w_rows, col_ptr)
    # A11 * B_top3
    col_ptr += add_strassen_2x2(a11_u_rows, b_top3_v_rows, c_sub3_w_rows, col_ptr)
    
    # A12 * B_bottom
    # B_bottom rows in V: j=2,3; k=0..5 -> 12 rows
    # B_bottom1: j=2,3; k=0,1 -> V rows [12, 13, 18, 19]
    b_bottom1_v_rows = [12, 13, 18, 19]
    
    # B_bottom2: j=2,3; k=2,3 -> V rows [14, 15, 20, 21]
    b_bottom2_v_rows = [14, 15, 20, 21]
    
    # B_bottom3: j=2,3; k=4,5 -> V rows [16, 17, 22, 23]
    b_bottom3_v_rows = [16, 17, 22, 23]
    
    # A12 * B_bottom1
    col_ptr += add_strassen_2x2(a12_u_rows, b_bottom1_v_rows, c_sub1_w_rows, col_ptr)
    # A12 * B_bottom2
    col_ptr += add_strassen_2x2(a12_u_rows, b_bottom2_v_rows, c_sub2_w_rows, col_ptr)
    # A12 * B_bottom3
    col_ptr += add_strassen_2x2(a12_u_rows, b_bottom3_v_rows, c_sub3_w_rows, col_ptr)
    
    # Part 2: A2 (1x4, last row of A) times B (4x6)
    # A2 is 1x4: row 2 of A, all 4 columns
    # In U: rows 8,9,10,11 (i=2, j=0,1,2,3 -> 2*4+j = 8,9,10,11)
    a2_u_rows = [8, 9, 10, 11]
    
    # C2 is 1x6: row 2 of C, all 6 columns
    # In W: rows 2,5,8,11,14,17 (k*n+i for i=2, k=0..5 -> k*3+2 = 2,5,8,11,14,17)
    c2_w_rows = [2, 5, 8, 11, 14, 17]
    
    # For 1x4 * 4x6, we have 6 dot products of length 4
    # Dot product k: sum_{j=0}^{3} A2[j] * B[j][k]
    # This is a rank-4 decomposition for each dot product
    
    for k in range(6):
        for j in range(4):
            # U: A2[j] is at row 8+j
            # V: B[j][k] is at row j*6+k
            # W: C2[k] is at row k*3+2
            U[8 + j][col_ptr] = 1
            V[j * 6 + k][col_ptr] = 1
            W[k * 3 + 2][col_ptr] = 1
            col_ptr += 1
    
    # Total should be 42 + 24 = 66
    assert col_ptr == total_rank, f"Expected {total_rank} columns, got {col_ptr}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
