"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Uses block decomposition:
- Split inner dimension (5) into 2 + 3
- For the 2-part: use Strassen-like rank-11 decomposition for 2x2x3 blocks
- For the 3-part: use schoolbook rank-18 for 2x3x3 blocks
- Total: 4 * 11 + 4 * 18 = 44 + 72 = 116 (vs schoolbook 120)
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns dict with U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    where rank is minimized.
    
    The decomposition satisfies:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise.
    
    Strategy: Block decomposition of the inner dimension.
    - Split j into {0,1} (size 2) and {2,3,4} (size 3)
    - For j in {0,1}: A is 4x2, B is 2x6
      * Split A into rows {0,1} and {2,3} (each 2x2)
      * Split B into cols {0,1,2} and {3,4,5} (each 2x3)
      * Each 2x2 * 2x3 -> 2x3 uses rank 11 (Strassen-based)
      * 4 such blocks: 4 * 11 = 44
    - For j in {2,3,4}: A is 4x3, B is 3x6
      * Split A into rows {0,1} and {2,3} (each 2x3)
      * Split B into cols {0,1,2} and {3,4,5} (each 3x3)
      * Each 2x3 * 3x3 -> 2x3 uses schoolbook rank 18
      * 4 such blocks: 4 * 18 = 72
    - Total rank: 44 + 72 = 116
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build U, V, W as lists of columns, then convert to matrices
    # Each "rank term" is a column in U, V, W
    u_cols = []
    v_cols = []
    w_cols = []
    
    def add_term(u_entry, v_entry, w_entry):
        """Add a rank-1 term to the decomposition."""
        u_cols.append(u_entry)
        v_cols.append(v_entry)
        w_cols.append(w_entry)
    
    def make_zero_u():
        return [0] * (n * m)  # 20 entries
    
    def make_zero_v():
        return [0] * (m * p)  # 30 entries
    
    def make_zero_w():
        return [0] * (p * n)  # 24 entries
    
    # ============================================================
    # Part 1: j in {0, 1} (inner dimension size 2)
    # A is 4x2 (rows 0-3, cols 0-1), B is 2x6 (rows 0-1, cols 0-5)
    # 
    # Split A into A_top (rows 0,1) and A_bot (rows 2,3), each 2x2
    # Split B into B_left (cols 0,1,2) and B_right (cols 3,4,5), each 2x3
    #
    # 4 independent 2x2 * 2x3 -> 2x3 multiplications:
    #   (A_top, B_left) -> C_top_left (2x3)
    #   (A_top, B_right) -> C_top_right (2x3)
    #   (A_bot, B_left) -> C_bot_left (2x3)
    #   (A_bot, B_right) -> C_bot_right (2x3)
    #
    # Each 2x2 * 2x3 uses rank 11.
    # ============================================================
    
    def strassen_2x2_x_2x3(a_row_offset, b_col_offset):
        """
        Add rank-11 decomposition for multiplying a 2x2 matrix by a 2x3 matrix.
        
        a_row_offset: starting row index in A (0 or 2)
        b_col_offset: starting col index in B (0 or 3)
        
        The 2x2 A block has entries A[a_row_offset + i, j] for i in {0,1}, j in {0,1}
        The 2x3 B block has entries B[j, b_col_offset + k] for j in {0,1}, k in {0,1,2}
        
        We use the following decomposition:
        For the 2x2 * 2x2 part (cols 0,1 of B), use Strassen rank 7.
        For the 2x2 * 2x1 part (col 2 of B), use schoolbook rank 4.
        Total: 7 + 4 = 11.
        
        Strassen for 2x2 * 2x2 (A_left 2x2, B_left 2x2 -> C_left 2x2):
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22) B11
        M3 = A11 (B12 - B22)
        M4 = A22 (B21 - B11)
        M5 = (A11 + A12) B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Schoolbook for 2x2 * 2x1 (A 2x2, B_col 2x1 -> C_col 2x1):
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        This is rank 4 (one term per (i,j,k) triple).
        """
        # Indices for A block: rows a_row_offset, a_row_offset+1; cols 0, 1 (in the 2-part)
        # But in the full A (4x5), cols are 0,1 for the j=0,1 part
        # So A[i, j] where i in {a_row_offset, a_row_offset+1}, j in {0, 1}
        
        # Indices for B block: rows 0, 1; cols b_col_offset, b_col_offset+1, b_col_offset+2
        # In the full B (5x6), rows are 0,1 for the j=0,1 part
        # So B[j, k] where j in {0, 1}, k in {b_col_offset, b_col_offset+1, b_col_offset+2}
        
        # Output C block: rows a_row_offset, a_row_offset+1; cols b_col_offset, b_col_offset+1, b_col_offset+2
        # In the full C (4x6), C[i, k] where i in {a_row_offset, a_row_offset+1}, k in {b_col_offset, ...}
        # W index: k * n + i = k * 4 + i
        
        ar = a_row_offset  # 0 or 2
        bc = b_col_offset  # 0 or 3
        
        # A block entries: A[ar + i, j] for i in {0,1}, j in {0,1}
        # In U index: (ar + i) * m + j = (ar + i) * 5 + j
        # B block entries: B[j, bc + k] for j in {0,1}, k in {0,1,2}
        # In V index: j * p + (bc + k) = j * 6 + bc + k
        # C block entries: C[ar + i, bc + k] for i in {0,1}, k in {0,1,2}
        # In W index: (bc + k) * n + (ar + i) = (bc + k) * 4 + ar + i
        
        # Strassen terms for 2x2 * 2x2 (first two cols of B)
        # A11 = A[ar, 0], A12 = A[ar, 1]
        # A21 = A[ar+1, 0], A22 = A[ar+1, 1]
        # B11 = B[0, bc], B12 = B[0, bc+1]
        # B21 = B[1, bc], B22 = B[1, bc+1]
        
        # M1 = (A11 + A22)(B11 + B22)
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 0] = 1       # A11
        u[(ar+1) * m + 1] = 1   # A22
        v[0 * p + bc] = 1       # B11
        v[1 * p + (bc+1)] = 1   # B22
        # This term contributes to C11 and C22
        # We need to track how this M1 contributes to output
        # Actually, for tensor decomposition, each rank-1 term is (u, v, w) where
        # T += u ⊗ v ⊗ w
        # So we need to express the product correctly.
        
        # Let me rethink. The Strassen algorithm computes C = A*B using 7 multiplications.
        # Each multiplication is a scalar product of two linear combinations.
        # In tensor form, each multiplication M_r = (sum_i a_i A[i]) * (sum_j b_j B[j])
        # contributes to output as: for each output element C[c] that uses M_r with coefficient w_c,
        # we add w_c * M_r to C[c].
        
        # So the rank-1 term for M1 is:
        # u has 1 at positions corresponding to A11 and A22
        # v has 1 at positions corresponding to B11 and B22
        # w has coefficients at positions corresponding to C11 and C22 (where M1 appears)
        
        # M1 appears in C11 with coeff +1 and C22 with coeff +1
        w[(bc + 0) * n + ar] = 1       # C11: row ar, col bc
        w[(bc + 1) * n + (ar+1)] = 1   # C22: row ar+1, col bc+1
        
        add_term(u, v, w)
        
        # M2 = (A21 + A22) B11
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[(ar+1) * m + 0] = 1   # A21
        u[(ar+1) * m + 1] = 1   # A22
        v[0 * p + bc] = 1       # B11
        # M2 appears in C21 with coeff +1
        w[(bc + 0) * n + (ar+1)] = 1  # C21: row ar+1, col bc
        add_term(u, v, w)
        
        # M3 = A11 (B12 - B22)
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 0] = 1       # A11
        v[0 * p + (bc+1)] = 1   # B12
        v[1 * p + (bc+1)] = -1  # -B22
        # M3 appears in C12 with coeff +1 and C22 with coeff +1
        w[(bc + 1) * n + ar] = 1       # C12: row ar, col bc+1
        w[(bc + 1) * n + (ar+1)] = 1   # C22: row ar+1, col bc+1
        add_term(u, v, w)
        
        # M4 = A22 (B21 - B11)
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[(ar+1) * m + 1] = 1   # A22
        v[1 * p + bc] = 1       # B21
        v[0 * p + bc] = -1      # -B11
        # M4 appears in C11 with coeff +1 and C21 with coeff +1
        w[(bc + 0) * n + ar] = 1       # C11: row ar, col bc
        w[(bc + 0) * n + (ar+1)] = 1   # C21: row ar+1, col bc
        add_term(u, v, w)
        
        # M5 = (A11 + A12) B22
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 0] = 1       # A11
        u[ar * m + 1] = 1       # A12
        v[1 * p + (bc+1)] = 1   # B22
        # M5 appears in C11 with coeff -1 and C12 with coeff +1
        w[(bc + 0) * n + ar] = -1      # C11: row ar, col bc
        w[(bc + 1) * n + ar] = 1       # C12: row ar, col bc+1
        add_term(u, v, w)
        
        # M6 = (A21 - A11)(B11 + B12)
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[(ar+1) * m + 0] = 1   # A21
        u[ar * m + 0] = -1      # -A11
        v[0 * p + bc] = 1       # B11
        v[0 * p + (bc+1)] = 1   # B12
        # M6 appears in C22 with coeff +1
        w[(bc + 1) * n + (ar+1)] = 1  # C22: row ar+1, col bc+1
        add_term(u, v, w)
        
        # M7 = (A12 - A22)(B21 + B22)
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 1] = 1       # A12
        u[(ar+1) * m + 1] = -1  # -A22
        v[1 * p + bc] = 1       # B21
        v[1 * p + (bc+1)] = 1   # B22
        # M7 appears in C11 with coeff +1
        w[(bc + 0) * n + ar] = 1  # C11: row ar, col bc
        add_term(u, v, w)
        
        # Now schoolbook for 2x2 * 2x1 (third col of B, index bc+2)
        # C1 = A11*B1 + A12*B2 where B1 = B[0, bc+2], B2 = B[1, bc+2]
        # C2 = A21*B1 + A22*B2
        
        # Term: A11 * B[0, bc+2]
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 0] = 1       # A11
        v[0 * p + (bc+2)] = 1   # B[0, bc+2]
        w[(bc + 2) * n + ar] = 1  # C1: row ar, col bc+2
        add_term(u, v, w)
        
        # Term: A12 * B[1, bc+2]
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[ar * m + 1] = 1       # A12
        v[1 * p + (bc+2)] = 1   # B[1, bc+2]
        w[(bc + 2) * n + ar] = 1  # C1: row ar, col bc+2
        add_term(u, v, w)
        
        # Term: A21 * B[0, bc+2]
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[(ar+1) * m + 0] = 1   # A21
        v[0 * p + (bc+2)] = 1   # B[0, bc+2]
        w[(bc + 2) * n + (ar+1)] = 1  # C2: row ar+1, col bc+2
        add_term(u, v, w)
        
        # Term: A22 * B[1, bc+2]
        u = make_zero_u()
        v = make_zero_v()
        w = make_zero_w()
        u[(ar+1) * m + 1] = 1   # A22
        v[1 * p + (bc+2)] = 1   # B[1, bc+2]
        w[(bc + 2) * n + (ar+1)] = 1  # C2: row ar+1, col bc+2
        add_term(u, v, w)
    
    # Apply Strassen 2x2 x 2x3 for the 4 blocks
    # A_top (rows 0,1) x B_left (cols 0,1,2)
    strassen_2x2_x_2x3(a_row_offset=0, b_col_offset=0)
    # A_top (rows 0,1) x B_right (cols 3,4,5)
    strassen_2x2_x_2x3(a_row_offset=0, b_col_offset=3)
    # A_bot (rows 2,3) x B_left (cols 0,1,2)
    strassen_2x2_x_2x3(a_row_offset=2, b_col_offset=0)
    # A_bot (rows 2,3) x B_right (cols 3,4,5)
    strassen_2x2_x_2x3(a_row_offset=2, b_col_offset=3)
    
    # ============================================================
    # Part 2: j in {2, 3, 4} (inner dimension size 3)
    # A is 4x3 (rows 0-3, cols 2-4), B is 3x6 (rows 2-4, cols 0-5)
    #
    # Split A into A_top (rows 0,1) and A_bot (rows 2,3), each 2x3
    # Split B into B_left (cols 0,1,2) and B_right (cols 3,4,5), each 3x3
    #
    # 4 independent 2x3 * 3x3 -> 2x3 multiplications:
    #   (A_top, B_left) -> C_top_left (2x3)
    #   (A_top, B_right) -> C_top_right (2x3)
    #   (A_bot, B_left) -> C_bot_left (2x3)
    #   (A_bot, B_right) -> C_bot_right (2x3)
    #
    # Each uses schoolbook rank 18 (2*3*3 = 18).
    # ============================================================
    
    def schoolbook_2x3_x_3x3(a_row_offset, b_col_offset):
        """
        Add schoolbook rank-18 decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
        
        a_row_offset: starting row index in A (0 or 2)
        b_col_offset: starting col index in B (0 or 3)
        
        A block: A[ar + i, j] for i in {0,1}, j in {2,3,4}
        B block: B[j, bc + k] for j in {2,3,4}, k in {0,1,2}
        C block: C[ar + i, bc + k] for i in {0,1}, k in {0,1,2}
        
        Schoolbook: for each (i, j, k), add term A[ar+i, j] * B[j, bc+k] to C[ar+i, bc+k]
        """
        ar = a_row_offset
        bc = b_col_offset
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    u = make_zero_u()
                    v = make_zero_v()
                    w = make_zero_w()
                    u[(ar + i) * m + (j + 2)] = 1  # A[ar+i, j+2]
                    v[(j + 2) * p + (bc + k)] = 1  # B[j+2, bc+k]
                    w[(bc + k) * n + (ar + i)] = 1  # C[ar+i, bc+k]
                    add_term(u, v, w)
    
    # Apply schoolbook 2x3 x 3x3 for the 4 blocks
    schoolbook_2x3_x_3x3(a_row_offset=0, b_col_offset=0)
    schoolbook_2x3_x_3x3(a_row_offset=0, b_col_offset=3)
    schoolbook_2x3_x_3x3(a_row_offset=2, b_col_offset=0)
    schoolbook_2x3_x_3x3(a_row_offset=2, b_col_offset=3)
    
    # Convert to matrices
    rank = len(u_cols)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for a in range(n * m):
            U[a, r] = u_cols[r][a]
        for b in range(m * p):
            V[b, r] = v_cols[r][b]
        for c in range(p * n):
            W[c, r] = w_cols[r][c]
    
    # Convert to lists of lists for the return format
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
