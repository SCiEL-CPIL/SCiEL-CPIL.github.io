"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Total rank is 60.
    # Blocks:
    # 1. B1 (3x2): Rank 17
    # 2. B2 (3x2): Rank 17
    # 3. B3 (3x3): Rank 26
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def add_block(U, V, W, offset_u, offset_v, offset_w):
        """
        Adds a sub-decomposition to the global matrices.
        U, V, W are local matrices for the sub-problem.
        offset_u, offset_v, offset_w are the starting indices in the global matrices.
        """
        rows_u, cols_u = U.shape
        rows_v, cols_v = V.shape
        rows_w, cols_w = W.shape
        assert cols_u == cols_v == cols_w, "Rank mismatch in block"
        rank = cols_u
        
        # We append to the list of blocks, handling offsets later
        # Or we can just accumulate into global arrays if we know the total size.
        # Since we are building from scratch, we can just store the blocks and concatenate.
        # But the indices in U, V, W are local to the block's A, B, C parts.
        # We need to map local indices to global indices.
        # This is complex. Instead, let's build the global matrices directly.
        pass

    # Global matrices
    total_rank = 60
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a scalar multiplication term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal current_rank
        U[u_idx][current_rank] += val_u
        V[v_idx][current_rank] += val_v
        W[w_idx][current_rank] += val_w
        current_rank += 1

    # Helper to add a Strassen 2x2x2 block
    def add_strassen_222(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Adds a 2x2x2 Strassen decomposition.
        A_rows, A_cols: indices of A's 2x2 block.
        B_rows, B_cols: indices of B's 2x2 block.
        C_rows, C_cols: indices of C's 2x2 block.
        """
        # Local indices for A: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
        # Global U index: i * m + j
        u00 = A_rows[0] * m + A_cols[0]
        u01 = A_rows[0] * m + A_cols[1]
        u10 = A_rows[1] * m + A_cols[0]
        u11 = A_rows[1] * m + A_cols[1]
        
        # Local indices for B: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
        # Global V index: j * p + k
        v00 = B_rows[0] * p + B_cols[0]
        v01 = B_rows[0] * p + B_cols[1]
        v10 = B_rows[1] * p + B_cols[0]
        v11 = B_rows[1] * p + B_cols[1]
        
        # Local indices for C: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
        # Global W index: k * n + i
        w00 = C_cols[0] * n + C_rows[0]
        w01 = C_cols[1] * n + C_rows[0]
        w10 = C_cols[0] * n + C_rows[1]
        w11 = C_cols[1] * n + C_rows[1]
        
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        add_term(u00, v00, w00, 1, 1, 1)
        add_term(u00, v00, w01, 1, 1, -1)
        add_term(u00, v00, w10, 1, 1, -1)
        add_term(u00, v00, w11, 1, 1, 1)
        add_term(u11, v00, w00, 1, 1, 1)
        add_term(u11, v00, w01, 1, 1, -1)
        add_term(u11, v00, w10, 1, 1, -1)
        add_term(u11, v00, w11, 1, 1, 1)
        add_term(u00, v11, w00, 1, 1, 1)
        add_term(u00, v11, w01, 1, 1, -1)
        add_term(u00, v11, w10, 1, 1, -1)
        add_term(u00, v11, w11, 1, 1, 1)
        add_term(u11, v11, w00, 1, 1, 1)
        add_term(u11, v11, w01, 1, 1, -1)
        add_term(u11, v11, w10, 1, 1, -1)
        add_term(u11, v11, w11, 1, 1, 1)
        
        # M2 = (A10 + A11)B00
        add_term(u10, v00, w00, 1, 1, 1)
        add_term(u10, v00, w10, 1, 1, 1)
        add_term(u11, v00, w00, 1, 1, 1)
        add_term(u11, v00, w10, 1, 1, 1)
        
        # M3 = A00(B01 - B11)
        add_term(u00, v01, w01, 1, 1, 1)
        add_term(u00, v01, w11, 1, 1, 1)
        add_term(u00, v11, w01, 1, 1, -1)
        add_term(u00, v11, w11, 1, 1, -1)
        
        # M4 = A11(B10 - B00)
        add_term(u11, v10, w10, 1, 1, 1)
        add_term(u11, v10, w11, 1, 1, 1)
        add_term(u11, v00, w10, 1, 1, -1)
        add_term(u11, v00, w11, 1, 1, -1)
        
        # M5 = (A00 + A01)B11
        add_term(u00, v11, w00, 1, 1, 1)
        add_term(u00, v11, w01, 1, 1, 1)
        add_term(u01, v11, w00, 1, 1, 1)
        add_term(u01, v11, w01, 1, 1, 1)
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(u10, v00, w00, 1, 1, 1)
        add_term(u10, v00, w10, 1, 1, 1)
        add_term(u10, v01, w00, 1, 1, 1)
        add_term(u10, v01, w10, 1, 1, 1)
        add_term(u00, v00, w00, 1, 1, -1)
        add_term(u00, v00, w10, 1, 1, -1)
        add_term(u00, v01, w00, 1, 1, -1)
        add_term(u00, v01, w10, 1, 1, -1)
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(u01, v10, w01, 1, 1, 1)
        add_term(u01, v10, w11, 1, 1, 1)
        add_term(u01, v11, w01, 1, 1, 1)
        add_term(u01, v11, w11, 1, 1, 1)
        add_term(u11, v10, w01, 1, 1, -1)
        add_term(u11, v10, w11, 1, 1, -1)
        add_term(u11, v11, w01, 1, 1, -1)
        add_term(u11, v11, w11, 1, 1, -1)

    # Helper for 2x1x2 (A is 2x1, B is 1x2, C is 2x2)
    def add_212(A_rows, A_col, B_row, B_cols, C_rows, C_cols):
        # A indices: (r1, c), (r2, c)
        u0 = A_rows[0] * m + A_col
        u1 = A_rows[1] * m + A_col
        # B indices: (r, c1), (r, c2)
        v0 = B_row * p + B_cols[0]
        v1 = B_row * p + B_cols[1]
        # C indices: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
        w00 = C_cols[0] * n + C_rows[0]
        w01 = C_cols[1] * n + C_rows[0]
        w10 = C_cols[0] * n + C_rows[1]
        w11 = C_cols[1] * n + C_rows[1]
        
        # Rank 4: A0*B0, A0*B1, A1*B0, A1*B1
        add_term(u0, v0, w00, 1, 1, 1)
        add_term(u0, v1, w01, 1, 1, 1)
        add_term(u1, v0, w10, 1, 1, 1)
        add_term(u1, v1, w11, 1, 1, 1)

    # Helper for 1x2x2 (A is 1x2, B is 2x2, C is 1x2)
    def add_122(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
        u0 = A_row * m + A_cols[0]
        u1 = A_row * m + A_cols[1]
        v00 = B_rows[0] * p + B_cols[0]
        v01 = B_rows[0] * p + B_cols[1]
        v10 = B_rows[1] * p + B_cols[0]
        v11 = B_rows[1] * p + B_cols[1]
        w0 = C_cols[0] * n + C_row
        w1 = C_cols[1] * n + C_row
        
        # Rank 4: A0*B00+A1*B10, A0*B01+A1*B11
        # Actually, C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        # We can do 4 terms:
        add_term(u0, v00, w0, 1, 1, 1)
        add_term(u1, v10, w0, 1, 1, 1)
        add_term(u0, v01, w1, 1, 1, 1)
        add_term(u1, v11, w1, 1, 1, 1)

    # Helper for 2x2x1 (A is 2x2, B is 2x1, C is 2x1)
    def add_221(A_rows, A_cols, B_rows, B_col, C_rows, C_col):
        u00 = A_rows[0] * m + A_cols[0]
        u01 = A_rows[0] * m + A_cols[1]
        u10 = A_rows[1] * m + A_cols[0]
        u11 = A_rows[1] * m + A_cols[1]
        v0 = B_rows[0] * p + B_col
        v1 = B_rows[1] * p + B_col
        w0 = C_col * n + C_rows[0]
        w1 = C_col * n + C_rows[1]
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        add_term(u00, v0, w0, 1, 1, 1)
        add_term(u01, v1, w0, 1, 1, 1)
        add_term(u10, v0, w1, 1, 1, 1)
        add_term(u11, v1, w1, 1, 1, 1)

    # Helper for 2x1x1
    def add_211(A_rows, A_col, B_row, B_col, C_rows, C_col):
        u0 = A_rows[0] * m + A_col
        u1 = A_rows[1] * m + A_col
        v0 = B_row * p + B_col
        w0 = C_col * n + C_rows[0]
        w1 = C_col * n + C_rows[1]
        add_term(u0, v0, w0, 1, 1, 1)
        add_term(u1, v0, w1, 1, 1, 1)

    # Helper for 1x2x1
    def add_121(A_row, A_cols, B_rows, B_col, C_row, C_col):
        u0 = A_row * m + A_cols[0]
        u1 = A_row * m + A_cols[1]
        v0 = B_rows[0] * p + B_col
        v1 = B_rows[1] * p + B_col
        w0 = C_col * n + C_row
        add_term(u0, v0, w0, 1, 1, 1)
        add_term(u1, v1, w0, 1, 1, 1)

    # Helper for 1x1x2
    def add_112(A_row, A_col, B_rows, B_cols, C_row, C_cols):
        u0 = A_row * m + A_col
        v0 = B_rows[0] * p + B_cols[0]
        v1 = B_rows[0] * p + B_cols[1]
        w0 = C_cols[0] * n + C_row
        w1 = C_cols[1] * n + C_row
        add_term(u0, v0, w0, 1, 1, 1)
        add_term(u0, v1, w1, 1, 1, 1)

    # Helper for 1x1x1
    def add_111(A_row, A_col, B_row, B_col, C_row, C_col):
        u0 = A_row * m + A_col
        v0 = B_row * p + B_col
        w0 = C_col * n + C_row
        add_term(u0, v0, w0, 1, 1, 1)

    # --- Block 1: B1 (cols 0, 1) ---
    # A11: rows 0,1; cols 0,1
    # a12: rows 0,1; col 2
    # a21T: row 2; cols 0,1
    # a22: row 2; col 2
    
    # b11: rows 0,1; cols 0,1
    # b21T: row 2; cols 0,1
    
    # C11: rows 0,1; cols 0,1
    # c12: rows 0,1; col 2 (but B1 has no col 2, so this is 0? No, C1 is 3x2)
    # C11 is 2x2.
    # C21 is 1x2.
    
    add_strassen_222([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_212([0, 1], 2, 2, [0, 1], [0, 1], [0, 1])
    add_122(2, [0, 1], [0, 1], [0, 1], 2, [0, 1])
    add_112(2, 2, 2, [0, 1], 2, [0, 1])
    
    # --- Block 2: B2 (cols 2, 3) ---
    add_strassen_222([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_212([0, 1], 2, 2, [2, 3], [0, 1], [2, 3])
    add_122(2, [0, 1], [0, 1], [2, 3], 2, [2, 3])
    add_112(2, 2, 2, [2, 3], 2, [2, 3])
    
    # --- Block 3: B3 (cols 4, 5, 6) ---
    # b11: rows 0,1; cols 4,5
    # b12: rows 0,1; col 6
    # b21T: row 2; cols 4,5
    # b22: row 2; col 6
    
    # C11: rows 0,1; cols 4,5
    # C12: rows 0,1; col 6
    # C21: row 2; cols 4,5
    # C22: row 2; col 6
    
    add_strassen_222([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5])
    add_212([0, 1], 2, 2, [4, 5], [0, 1], [4, 5])
    add_221([0, 1], [0, 1], [0, 1], 6, [0, 1], 6)
    add_211([0, 1], 2, 2, 6, [0, 1], 6)
    add_122(2, [0, 1], [0, 1], [4, 5], 2, [4, 5])
    add_121(2, [0, 1], [0, 1], 6, 2, 6)
    add_112(2, 2, 2, [4, 5], 2, [4, 5])
    add_111(2, 2, 2, 6, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
