import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        # A_idx: [a, b, c, d] indices in U (2x2 block)
        # B_idx: [e, f, g, h] indices in V (2x2 block)
        # C_idx: [c11, c12, c21, c22] indices in W (2x2 block)
        
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    def add_dot_block(A_idx, B_idx, C_idx, start_col):
        # A_idx: list of 4 indices in U for A[:, j] (actually rows 0 and 1 for col j)
        # Wait, A is 2x4. B is 4x1.
        # We process one column of B at a time? No, we process the whole 4x1 block.
        # The block is B[:, col_offset]
        # A_idx should be the indices for A.
        # Let's say we want to compute C[:, col_offset] = A @ B[:, col_offset]
        # C has 2 rows.
        # For each j in 0..3:
        #   U[A_row0, j][k] = 1
        #   U[A_row1, j][k] = 1
        #   V[B_row_j, col_offset][k] = 1
        #   W[C_col_offset, 0][k] = 1
        #   W[C_col_offset, 1][k] = 1
        
        # A_idx: [idx for A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
        # Actually, let's just pass the column offset in B and the column index in C.
        # And we know A is fixed.
        
        b_col = B_idx # This is the column index in B (0..5)
        c_col = C_idx # This is the column index in C (0..5)
        
        for j in range(4):
            # A[0, j] index in U: 0*4 + j = j
            # A[1, j] index in U: 1*4 + j = 4 + j
            # B[j, b_col] index in V: j*6 + b_col
            # C[c_col, 0] index in W: c_col*2 + 0
            # C[c_col, 1] index in W: c_col*2 + 1
            
            U[j][start_col] += 1
            U[4+j][start_col] += 1
            V[j*6 + b_col][start_col] += 1
            W[c_col*2 + 0][start_col] += 1
            W[c_col*2 + 1][start_col] += 1
            start_col += 1
            
        return start_col

    # Main loop
    # We split B into two 4x3 blocks.
    # Block 1: cols 0, 1, 2 of B -> cols 0, 1, 2 of C
    # Block 2: cols 3, 4, 5 of B -> cols 3, 4, 5 of C
    
    for block in range(2):
        b_start = block * 3
        c_start = block * 3
        
        # Part 1: B_L (4x2) -> C_L (2x2)
        # B cols: b_start, b_start+1
        # C cols: c_start, c_start+1
        
        # A is 2x4. We split A into A1 (cols 0,1) and A2 (cols 2,3).
        # A1 indices: 0, 1, 4, 5
        # A2 indices: 2, 3, 6, 7
        
        # B_L is 4x2. We split B_L into B1 (rows 0,1) and B2 (rows 2,3).
        # B1 indices in V:
        #   row 0: 0*6 + b_start, 0*6 + b_start+1
        #   row 1: 1*6 + b_start, 1*6 + b_start+1
        # B2 indices in V:
        #   row 2: 2*6 + b_start, 2*6 + b_start+1
        #   row 3: 3*6 + b_start, 3*6 + b_start+1
        
        # C_L is 2x2.
        # C11: c_start, 0 -> c_start*2 + 0
        # C12: c_start+1, 0 -> (c_start+1)*2 + 0
        # C21: c_start, 1 -> c_start*2 + 1
        # C22: c_start+1, 1 -> (c_start+1)*2 + 1
        
        A1_idx = [0, 1, 4, 5]
        B1_idx = [0*6 + b_start, 0*6 + b_start+1, 1*6 + b_start, 1*6 + b_start+1]
        C_L_idx = [c_start*2 + 0, (c_start+1)*2 + 0, c_start*2 + 1, (c_start+1)*2 + 1]
        
        col = add_strassen_block(A1_idx, B1_idx, C_L_idx, col)
        
        A2_idx = [2, 3, 6, 7]
        B2_idx = [2*6 + b_start, 2*6 + b_start+1, 3*6 + b_start, 3*6 + b_start+1]
        # C_L_idx is the same
        
        col = add_strassen_block(A2_idx, B2_idx, C_L_idx, col)
        
        # Part 2: B_R (4x1) -> C_R (2x1)
        # B col: b_start+2
        # C col: c_start+2
        
        col = add_dot_block(None, b_start+2, c_start+2, col)

    return {"U": U, "V": V, "W": W}
