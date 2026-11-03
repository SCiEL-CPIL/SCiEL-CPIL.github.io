"""Matrix multiplication 3x5x7 rank-minimal tensor decomposition."""
import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5x7 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on 2x2x2 blocks where possible,
    combined with schoolbook for remaining parts.
    
    For 3x5x7:
    - We can partition the 3x5 matrix into blocks and the 5x7 matrix into blocks
    - Use Strassen (rank 7 instead of 8) for 2x2x2 blocks
    - Use schoolbook (rank 1) for 1x1x1 blocks
    
    Partition strategy:
    - A is 3x5, B is 5x7, C is 3x7
    - Split A into rows: A1 (2x5), A2 (1x5)
    - Split B into columns: B1 (5x6), B2 (5x1)
    
    For A1*B1 (2x5 * 5x6 -> 2x6):
      Split A1 into A1a (2x3), A1b (2x2)
      Split B1 into B1a (3x6), B1b (2x6)
      A1*B1 = A1a*B1a + A1b*B1b
      
      A1a*B1a (2x3 * 3x6 -> 2x6):
        Split B1a into B1a1 (3x4), B1a2 (3x2)
        A1a*B1a = A1a*B1a1 + A1a*B1a2
        A1a*B1a1 (2x3 * 3x4 -> 2x4):
          Use Strassen: split into 2x2 blocks where possible
          A1a = [[a11, a12, a13], [a21, a22, a23]]
          B1a1 = [[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34]]
          This is 2x3 * 3x4. We can do:
          Split A1a into A1a_left (2x2), A1a_right (2x1)
          Split B1a1 into B1a1_top (2x4), B1a1_bottom (1x4)
          A1a*B1a1 = A1a_left*B1a1_top + A1a_right*B1a1_bottom
          A1a_left*B1a1_top (2x2 * 2x4 -> 2x4): Use Strassen (rank 7)
          A1a_right*B1a1_bottom (2x1 * 1x4 -> 2x4): Schoolbook (rank 2*1*4=8? No, it's 2*1*4 but actually rank = 2*1*4 = 8? Wait, for 2x1 * 1x4, the tensor has 2*1*4 = 8 entries but rank is 2*4 = 8? Actually for matrix multiplication (2,1,4), rank = 2*1*4 = 8. But we can do better: this is just outer product style. Actually (2x1)*(1x4) = 2x4, each entry C[i,k] = A[i,0]*B[0,k]. So rank = 2*4 = 8? No wait, that's the schoolbook. But actually this is a rank-2 tensor? No. Let me reconsider.
          
          Actually, let me just use a simpler approach. Let me directly compute the rank.
          
          For 2x3 * 3x4 -> 2x4:
          Schoolbook rank = 2*3*4 = 24
          
          Let me try a different partitioning that's cleaner.
          
    Let me use a more systematic approach. I'll use the known result that for small matrix multiplication, we can use Strassen recursively.
    
    Actually, let me just directly construct a good decomposition by hand.
    
    For 3x5x7, the schoolbook rank is 3*5*7 = 105.
    
    Using Strassen on 2x2x2 subproblems:
    - 2x2x2 has rank 7 instead of 8, saving 1 per block.
    
    Let me partition:
    A (3x5) = [A_top (2x5); A_bottom (1x5)]
    B (5x7) = [B_left (5x6) | B_right (5x1)]
    
    C = A*B = [A_top*B_left + A_top*B_right; A_bottom*B_left + A_bottom*B_right]
    
    A_top*B_left (2x5 * 5x6 -> 2x6):
      A_top = [A_tl (2x3) | A_tr (2x2)]
      B_left = [B_tl (3x6); B_bl (2x6)]
      A_top*B_left = A_tl*B_tl + A_tr*B_bl
      
      A_tl*B_tl (2x3 * 3x6 -> 2x6):
        A_tl = [A_tll (2x2) | A_tlr (2x1)]
        B_tl = [B_tll (2x6); B_blr (1x6)]
        A_tl*B_tl = A_tll*B_tll + A_tlr*B_blr
        
        A_tll*B_tll (2x2 * 2x6 -> 2x6):
          Split B_tll into 3 blocks of 2x2: B_tll = [B1 | B2 | B3]
          A_tll*B_tll = A_tll*B1 + A_tll*B2 + A_tll*B3
          Each A_tll*Bk is 2x2 * 2x2 -> 2x2, rank 7 with Strassen.
          Total: 3*7 = 21
        
        A_tlr*B_blr (2x1 * 1x6 -> 2x6):
          Schoolbook: 2*1*6 = 12
        
      A_tr*B_bl (2x2 * 2x6 -> 2x6):
        Split B_bl into 3 blocks of 2x2: B_bl = [C1 | C2 | C3]
        A_tr*Ck each is 2x2 * 2x2 -> 2x2, rank 7.
        Total: 3*7 = 21
      
    A_top*B_right (2x5 * 5x1 -> 2x1):
      A_top = [A_tl2 (2x3) | A_tr2 (2x2)]
      B_right = [B_tr (3x1); B_br (2x1)]
      A_top*B_right = A_tl2*B_tr + A_tr2*B_br
      
      A_tl2*B_tr (2x3 * 3x1 -> 2x1):
        A_tl2 = [A_tll2 (2x2) | A_tlr2 (2x1)]
        B_tr = [B_tll2 (2x1); B_blr2 (1x1)]
        A_tl2*B_tr = A_tll2*B_tll2 + A_tlr2*B_blr2
        
        A_tll2*B_tll2 (2x2 * 2x1 -> 2x1):
          Schoolbook: 2*2*1 = 4
          
        A_tlr2*B_blr2 (2x1 * 1x1 -> 2x1):
          Schoolbook: 2*1*1 = 2
        
      A_tr2*B_br (2x2 * 2x1 -> 2x1):
        Schoolbook: 2*2*1 = 4
    
    A_bottom*B_left (1x5 * 5x6 -> 1x6):
      A_bottom = [A_bl (1x3) | A_br (1x2)]
      B_left = [B_tl2 (3x6); B_bl2 (2x6)]
      A_bottom*B_left = A_bl*B_tl2 + A_br*B_bl2
      
      A_bl*B_tl2 (1x3 * 3x6 -> 1x6):
        A_bl = [A_bll (1x2) | A_blr (1x1)]
        B_tl2 = [B_tll2 (2x6); B_blr2 (1x6)]
        A_bl*B_tl2 = A_bll*B_tll2 + A_blr*B_blr2
        
        A_bll*B_tll2 (1x2 * 2x6 -> 1x6):
          Schoolbook: 1*2*6 = 12
          
        A_blr*B_blr2 (1x1 * 1x6 -> 1x6):
          Schoolbook: 1*1*6 = 6
        
      A_br*B_bl2 (1x2 * 2x6 -> 1x6):
        Schoolbook: 1*2*6 = 12
    
    A_bottom*B_right (1x5 * 5x1 -> 1x1):
      A_bottom = [A_bl3 (1x3) | A_br3 (1x2)]
      B_right = [B_tr3 (3x1); B_br3 (2x1)]
      A_bottom*B_right = A_bl3*B_tr3 + A_br3*B_br3
      
      A_bl3*B_tr3 (1x3 * 3x1 -> 1x1):
        Schoolbook: 1*3*1 = 3
        
      A_br3*B_br3 (1x2 * 2x1 -> 1x1):
        Schoolbook: 1*2*1 = 2
    
    Total rank = 21 + 12 + 21 + 4 + 2 + 4 + 12 + 6 + 12 + 3 + 2 = 99
    
    Hmm, that gives 99. Let me see if I can do better.
    
    Actually, the known lower bound for 3x5x7 is not well established, but 99 seems reasonable.
    The seed program uses rank 105 (schoolbook). The inspirations mention rank 99.
    
    Let me just implement the rank 99 decomposition as described above.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build the U, V, W matrices incrementally
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (35, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        """Add a single rank-1 term."""
        U_rows.append((u_idx, u_val))
        V_rows.append((v_idx, v_val))
        W_rows.append((w_idx, w_val))
    
    def add_strassen_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        a: A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        b: B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        c: C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        
        Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # Indices in A: rows a_row_offset, a_row_offset+1; cols a_col_offset, a_col_offset+1
        # Indices in B: rows b_row_offset, b_row_offset+1; cols b_col_offset, b_col_offset+1
        # Indices in C: rows c_row_offset, c_row_offset+1; cols c_col_offset, c_col_offset+1
        
        a00 = (a_row_offset, a_col_offset)
        a01 = (a_row_offset, a_col_offset + 1)
        a10 = (a_row_offset + 1, a_col_offset)
        a11 = (a_row_offset + 1, a_col_offset + 1)
        
        b00 = (b_row_offset, b_col_offset)
        b01 = (b_row_offset, b_col_offset + 1)
        b10 = (b_row_offset + 1, b_col_offset)
        b11 = (b_row_offset + 1, b_col_offset + 1)
        
        c00 = (c_row_offset, c_col_offset)
        c01 = (c_row_offset, c_col_offset + 1)
        c10 = (c_row_offset + 1, c_col_offset)
        c11 = (c_row_offset + 1, c_col_offset + 1)
        
        def a_idx(i, j):
            return i * m + j
        
        def b_idx(i, j):
            return i * p + j
        
        def c_idx(i, j):
            return j * n + i  # W[k*n+i, r] where k is col of C, i is row of C
        
        # M1 = (A00+A11)*(B00+B11)
        # U: a00 + a11, V: b00 + b11, W: c00 + c11
        add_term_combined([a_idx(a00[0], a00[1]), a_idx(a11[0], a11[1])],
                         [b_idx(b00[0], b00[1]), b_idx(b11[0], b11[1])],
                         [c_idx(c00[0], c00[1]), c_idx(c11[0], c11[1])])
        
        # M2 = (A10+A11)*B00
        add_term_combined([a_idx(a10[0], a10[1]), a_idx(a11[0], a11[1])],
                         [b_idx(b00[0], b00[1])],
                         [c_idx(c10[0], c10[1])])
        
        # M3 = A00*(B01-B11)
        add_term_combined([a_idx(a00[0], a00[1])],
                         [b_idx(b01[0], b01[1]), b_idx(b11[0], b11[1])],
                         [c_idx(c01[0], c01[1])])
        # Need to handle the subtraction in V
        # Let me redo this more carefully
        
    # Let me redo this with a cleaner approach
    U_rows = []
    V_rows = []
    W_rows = []
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(i, j):
        return i * p + j
    
    def c_idx(i, j):
        return j * n + i
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Strassen for 2x2*2x2->2x2 starting at given offsets."""
        # M1 = (A00+A11)*(B00+B11)
        U_rows.append([(a_idx(a_r0, a_c0), 1), (a_idx(a_r0+1, a_c0+1), 1)])
        V_rows.append([(b_idx(b_r0, b_c0), 1), (b_idx(b_r0+1, b_c0+1), 1)])
        W_rows.append([(c_idx(c_r0, c_c0), 1), (c_idx(c_r0+1, c_c0+1), 1)])
        
        # M2 = (A10+A11)*B00
        U_rows.append([(a_idx(a_r0+1, a_c0), 1), (a_idx(a_r0+1, a_c0+1), 1)])
        V_rows.append([(b_idx(b_r0, b_c0), 1)])
        W_rows.append([(c_idx(c_r0+1, c_c0), 1)])
        
        # M3 = A00*(B01-B11)
        U_rows.append([(a_idx(a_r0, a_c0), 1)])
        V_rows.append([(b_idx(b_r0, b_c0+1), 1), (b_idx(b_r0+1, b_c0+1), -1)])
        W_rows.append([(c_idx(c_r0, c_c0+1), 1)])
        
        # M4 = A11*(B10-B00)
        U_rows.append([(a_idx(a_r0+1, a_c0+1), 1)])
        V_rows.append([(b_idx(b_r0+1, b_c0), 1), (b_idx(b_r0, b_c0), -1)])
        W_rows.append([(c_idx(c_r0, c_c0), 1), (c_idx(c_r0+1, c_c0+1), 1)])
        
        # M5 = (A00+A01)*B11
        U_rows.append([(a_idx(a_r0, a_c0), 1), (a_idx(a_r0, a_c0+1), 1)])
        V_rows.append([(b_idx(b_r0+1, b_c0+1), 1)])
        W_rows.append([(c_idx(c_r0, c_c0), -1), (c_idx(c_r0, c_c0+1), 1)])
        
        # M6 = (A10-A00)*(B00+B01)
        U_rows.append([(a_idx(a_r0+1, a_c0), 1), (a_idx(a_r0, a_c0), -1)])
        V_rows.append([(b_idx(b_r0, b_c0), 1), (b_idx(b_r0, b_c0+1), 1)])
        W_rows.append([(c_idx(c_r0+1, c_c0+1), 1)])
        
        # M7 = (A01-A11)*(B10+B11)
        U_rows.append([(a_idx(a_r0, a_c0+1), 1), (a_idx(a_r0+1, a_c0+1), -1)])
        V_rows.append([(b_idx(b_r0+1, b_c0), 1), (b_idx(b_r0+1, b_c0+1), 1)])
        W_rows.append([(c_idx(c_r0, c_c0), 1), (c_idx(c_r0+1, c_c0+1), -1)])
    
    def add_schoolbook(a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1):
        """Schoolbook for submatrix multiplication."""
        for i in range(a_r0, a_r1):
            for j in range(a_c0, a_c1):
                for k in range(b_c0, b_c1):
                    U_rows.append([(a_idx(i, j), 1)])
                    V_rows.append([(b_idx(j, k), 1)])
                    W_rows.append([(c_idx(i, k), 1)])
    
    # Now build the full decomposition
    # A is 3x5, B is 5x7, C is 3x7
    
    # Partition:
    # A = [A_top (2x5); A_bottom (1x5)]
    # B = [B_left (5x6) | B_right (5x1)]
    
    # A_top*B_left (2x5 * 5x6 -> 2x6):
    #   A_top = [A_tl (2x3) | A_tr (2x2)]
    #   B_left = [B_tl (3x6); B_bl (2x6)]
    #   A_top*B_left = A_tl*B_tl + A_tr*B_bl
    
    # A_tl*B_tl (2x3 * 3x6 -> 2x6):
    #   A_tl = [A_tll (2x2) | A_tlr (2x1)]
    #   B_tl = [B_tll (2x6); B_blr (1x6)]
    #   A_tl*B_tl = A_tll*B_tll + A_tlr*B_blr
    
    # A_tll*B_tll (2x2 * 2x6 -> 2x6): 3 Strassen blocks
    add_strassen(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 0, 0, 4, 0, 4)  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    
    # A_tlr*B_blr (2x1 * 1x6 -> 2x6): schoolbook
    add_schoolbook(0, 2, 2, 3, 2, 0, 3, 6)
    
    # A_tr*B_bl (2x2 * 2x6 -> 2x6): 3 Strassen blocks
    add_strassen(0, 3, 3, 0, 0, 0)  # A[0:2,3:5] * B[3:5,0:2] -> C[0:2,0:2]
    add_strassen(0, 3, 3, 2, 0, 2)  # A[0:2,3:5] * B[3:5,2:4] -> C[0:2,2:4]
    add_strassen(0, 3, 3, 4, 0, 4)  # A[0:2,3:5] * B[3:5,4:6] -> C[0:2,4:6]
    
    # A_top*B_right (2x5 * 5x1 -> 2x1):
    #   A_top = [A_tl2 (2x3) | A_tr2 (2x2)]
    #   B_right = [B_tr (3x1); B_br (2x1)]
    #   A_top*B_right = A_tl2*B_tr + A_tr2*B_br
    
    # A_tl2*B_tr (2x3 * 3x1 -> 2x1):
    #   A_tl2 = [A_tll2 (2x2) | A_tlr2 (2x1)]
    #   B_tr = [B_tll2 (2x1); B_blr2 (1x1)]
    #   A_tl2*B_tr = A_tll2*B_tll2 + A_tlr2*B_blr2
    
    # A_tll2*B_tll2 (2x2 * 2x1 -> 2x1): schoolbook
    add_schoolbook(0, 0, 2, 2, 0, 6, 2, 7)
    
    # A_tlr2*B_blr2 (2x1 * 1x1 -> 2x1): schoolbook
    add_schoolbook(0, 2, 2, 3, 2, 6, 3, 7)
    
    # A_tr2*B_br (2x2 * 2x1 -> 2x1): schoolbook
    add_schoolbook(0, 3, 2, 5, 3, 6, 5, 7)
    
    # A_bottom*B_left (1x5 * 5x6 -> 1x6):
    #   A_bottom = [A_bl (1x3) | A_br (1x2)]
    #   B_left = [B_tl2 (3x6); B_bl2 (2x6)]
    #   A_bottom*B_left = A_bl*B_tl2 + A_br*B_bl2
    
    # A_bl*B_tl2 (1x3 * 3x6 -> 1x6):
    #   A_bl = [A_bll (1x2) | A_blr (1x1)]
    #   B_tl2 = [B_tll2 (2x6); B_blr2 (1x6)]
    #   A_bl*B_tl2 = A_bll*B_tll2 + A_blr*B_blr2
    
    # A_bll*B_tll2 (1x2 * 2x6 -> 1x6): schoolbook
    add_schoolbook(2, 0, 3, 2, 0, 0, 2, 6)
    
    # A_blr*B_blr2 (1x1 * 1x6 -> 1x6): schoolbook
    add_schoolbook(2, 2, 3, 3, 2, 0, 3, 6)
    
    # A_br*B_bl2 (1x2 * 2x6 -> 1x6): schoolbook
    add_schoolbook(2, 3, 3, 5, 3, 0, 5, 6)
    
    # A_bottom*B_right (1x5 * 5x1 -> 1x1):
    #   A_bottom = [A_bl3 (1x3) | A_br3 (1x2)]
    #   B_right = [B_tr3 (3x1); B_br3 (2x1)]
    #   A_bottom*B_right = A_bl3*B_tr3 + A_br3*B_br3
    
    # A_bl3*B_tr3 (1x3 * 3x1 -> 1x1): schoolbook
    add_schoolbook(2, 0, 3, 3, 0, 6, 3, 7)
    
    # A_br3*B_br3 (1x2 * 2x1 -> 1x1): schoolbook
    add_schoolbook(2, 3, 3, 5, 3, 6, 5, 7)
    
    # Now convert to dense matrices
    rank = len(U_rows)
    nU = n * m
    nV = m * p
    nW = p * n
    
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    
    for r in range(rank):
        for idx, val in U_rows[r]:
            U[idx][r] += val
        for idx, val in V_rows[r]:
            V[idx][r] += val
        for idx, val in W_rows[r]:
            W[idx][r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
        for i in range(n):
            print(f"Row {i}: expected {C[i]}, got {C_decomp[i]}", file=sys.stderr)
    else:
        print(f"Verification passed. Rank = {rank}", file=sys.stderr)
