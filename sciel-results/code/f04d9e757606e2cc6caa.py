"""Matrix multiplication 3,4,7 with subproduct + Strassen decomposition."""
DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix mult with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U rows: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V rows: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W rows: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U row 0=1, row 3=1; V row 0=1, row 3=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # W: C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: U row 2=1, row 3=1; V row 0=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # W: C21 += M2, C22 -= M2
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h): U row 0=1; V row 1=1, row 3=-1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # W: C12 += M3, C22 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e): U row 3=1; V row 2=1, row 0=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # W: C11 += M4, C21 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h: U row 0=1, row 1=1; V row 3=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # W: C11 -= M5, C12 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-d)(f+h): U row 2=1, row 3=-1; V row 1=1, row 3=1
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    # W: C22 += M6
    W[3][5] = 1
    
    # M7 = (b-d)(e+g): U row 1=1, row 3=-1; V row 0=1, row 2=1
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = 1
    # W: C11 += M7
    W[0][6] = 1
    
    return U, V, W

def _strassen_2x2x1():
    """Return (U, V, W) for 2x2x1 matrix mult with rank 4.
    A is 2x2, B is 2x1, C is 2x1.
    U: (4, 4), V: (2, 4), W: (2, 4)
    U[i*2+j, c], V[j*1+k, c]=V[j, c], W[k*2+i, c]
    """
    # C = A*B where A=[[a,b],[c,d]], B=[e,f], C=[c1,c2]
    # c1 = ae+bf, c2 = ce+df
    # Use 4 multiplications (schoolbook for 2x2x1)
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # M1 = a*e
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # M2 = b*f
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    # M3 = c*e
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    # M4 = d*f
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 3x4 * 4x7 matrix multiplication.
    
    Strategy: Split the 3x4 input into 2x4 (top) and 1x4 (bottom) blocks.
    Split the 4x7 input into 4x3 (left) and 4x4 (right) blocks.
    
    Sub-tensors:
    - 2x4x3: decompose as 2x(2x2)x3 = two 2x2x2 Strassen blocks (rank 7 each) = 14
    - 2x4x4: decompose as 2x(2x2)x4 = four 2x2x2 Strassen blocks (rank 7 each) = 28
    - 1x4x7: schoolbook, rank 28
    
    Wait, let me reconsider. The output is 3x7.
    
    Actually, let me think more carefully:
    - Input A is 3x4, Input B is 4x7, Output C is 3x7.
    - Split A into A_top (2x4) and A_bot (1x4).
    - Split B into B_left (4x3) and B_right (4x4).
    
    C_top_left = A_top * B_left: 2x3 output, rank via 2x4x3
    C_top_right = A_top * B_right: 2x4 output, rank via 2x4x4  
    C_bot_left = A_bot * B_left: 1x3 output, rank via 1x4x3
    C_bot_right = A_bot * B_right: 1x4 output, rank via 1x4x4
    
    For 2x4x3: Split the 4 columns of A into two 2-col groups, and 3 cols of B.
    This gives 2*(2x2x3) sub-tensors. 2x2x3 can be done with rank 6 (or use Strassen-like).
    Actually 2x2x3: A is 2x2, B is 2x3, C is 2x3. Rank is 6 (schoolbook: 2*2*3=12, but we can do better).
    Actually for 2x2x3, the tensor rank is 6. Let me just use schoolbook for small ones.
    
    Let me use a cleaner approach:
    - 2x4x7: Use Strassen on 2x2 blocks. Split 4 into 2+2, 7 into 2+2+2+1.
      That's 2 * 4 = 8 blocks of 2x2x{2 or 1}.
      2x2x2 blocks: 7 each, 2x2x1 blocks: 4 each.
      6 blocks of 2x2x2 (for 2*3=6 of the 7 cols... wait 7=2+2+2+1, so 3 blocks of 2x2x2 and 1 block of 2x2x1 per row-pair)
      Actually: 2x4x7 = sum over (i in {0,1}, j in {0,1,2,3}, k in {0,...,6})
      Group by 2x2 blocks of A (2 groups in col direction) and 2x2 or 2x1 blocks of B.
      A has 4 cols -> 2 groups of 2. B has 7 cols -> 3 groups of 2 + 1 group of 1.
      So: 2 * 3 = 6 blocks of 2x2x2 (rank 7 each) + 2 * 1 = 2 blocks of 2x2x1 (rank 4 each)
      Total for 2x4x7: 6*7 + 2*4 = 42 + 8 = 50
      
    - 1x4x7: schoolbook, rank 28
    
    Total: 50 + 28 = 78
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m+j, c] for i in range(n), j in range(m)
    # V[j*p+k, c] for j in range(m), k in range(p)
    # W[k*n+i, c] for k in range(p), i in range(n)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_block(u_mat, v_mat, w_mat, row_offset_u, row_offset_v, row_offset_w):
        """Add a block decomposition. u_mat is (ru, rc), v_mat is (rv, rc), w_mat is (rw, rc).
        row_offset_u, row_offset_v, row_offset_w are the starting row indices in U, V, W.
        """
        rc = len(u_mat[0]) if u_mat else 0
        for c in range(rc):
            U_rows.append([0] * rc)
            V_rows.append([0] * rc)
            W_rows.append([0] * rc)
            for i in range(len(u_mat)):
                U_rows[-1][c] = 0  # placeholder, will fill below
            # Actually let me restructure - I'll collect all contributions
        # This approach is getting complicated. Let me use a different method.
        pass
    
    # Better approach: collect all (row_u, col, val), (row_v, col, val), (row_w, col, val) tuples
    contributions_u = {}  # (row, col) -> value
    contributions_v = {}
    contributions_w = {}
    current_rank = 0
    
    def add_decomposition(u_mat, v_mat, w_mat, u_offset, v_offset, w_offset):
        """u_mat: list of lists, shape (ru, rc)
           v_mat: list of lists, shape (rv, rc)
           w_mat: list of lists, shape (rw, rc)
           u_offset, v_offset, w_offset: row offsets in the full U, V, W matrices
        """
        nonlocal current_rank
        rc = len(u_mat[0]) if u_mat else 0
        for c in range(rc):
            col = current_rank + c
            for i in range(len(u_mat)):
                if u_mat[i][c] != 0:
                    key = (u_offset + i, col)
                    contributions_u[key] = contributions_u.get(key, 0) + u_mat[i][c]
            for j in range(len(v_mat)):
                if v_mat[j][c] != 0:
                    key = (v_offset + j, col)
                    contributions_v[key] = contributions_v.get(key, 0) + v_mat[j][c]
            for k in range(len(w_mat)):
                if w_mat[k][c] != 0:
                    key = (w_offset + k, col)
                    contributions_w[key] = contributions_w.get(key, 0) + w_mat[k][c]
        current_rank += rc
    
    # 2x4x7 sub-tensor: top 2 rows of A (rows 0,1), all 4 cols, all 7 cols of B
    # Split A's 4 cols into 2 groups of 2: cols (0,1) and (2,3)
    # Split B's 7 cols into groups: (0,1), (2,3), (4,5), (6)
    # For each A-col-group and B-col-group, we have a 2x2x{2 or 1} sub-tensor
    
    for a_col_group in range(2):  # 2 groups of 2 cols in A
        a_col_offset = a_col_group * 2
        for b_col_group in range(4):  # 3 groups of 2 + 1 group of 1 in B
            if b_col_group < 3:
                b_col_offset = b_col_group * 2
                b_width = 2
                # 2x2x2 Strassen
                u_s, v_s, w_s = _strassen_2x2x2()
                # U rows: i in {0,1}, j in {0,1} -> row = i*2+j, offset by a_col_offset
                # V rows: j in {0,1}, k in {0,1} -> row = j*2+k, offset by b_col_offset
                # W rows: k in {0,1}, i in {0,1} -> row = k*2+i, offset by b_col_offset (for k) and 0 (for i, since top 2 rows)
                add_decomposition(u_s, v_s, w_s, a_col_offset, b_col_offset, b_col_offset)
            else:
                b_col_offset = 6
                b_width = 1
                # 2x2x1: A is 2x2, B is 2x1, C is 2x1
                u_s, v_s, w_s = _strassen_2x2x1()
                # U rows: i in {0,1}, j in {0,1} -> row = i*2+j, offset by a_col_offset
                # V rows: j in {0,1}, k in {0} -> row = j*1+0 = j, offset by b_col_offset
                # W rows: k in {0}, i in {0,1} -> row = k*2+i = i, offset by b_col_offset (for k=0) and 0 (for i)
                add_decomposition(u_s, v_s, w_s, a_col_offset, b_col_offset, b_col_offset)
    
    # 1x4x7 sub-tensor: bottom row of A (row 2), all 4 cols, all 7 cols of B
    # Schoolbook: 1*4*7 = 28 multiplications
    for j in range(4):
        for k in range(7):
            u_mat = [[1]]
            v_mat = [[1]]
            w_mat = [[1]]
            # U: row = 2*4+j = 8+j
            # V: row = j*7+k
            # W: row = k*3+2
            add_decomposition(u_mat, v_mat, w_mat, 8 + j, j * 7 + k, k * 3 + 2)
    
    # Now build the full matrices
    U = [[0] * current_rank for _ in range(n * m)]
    V = [[0] * current_rank for _ in range(m * p)]
    W = [[0] * current_rank for _ in range(p * n)]
    
    for (row, col), val in contributions_u.items():
        U[row][col] = val
    for (row, col), val in contributions_v.items():
        V[row][col] = val
    for (row, col), val in contributions_w.items():
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
