import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank 78
    rank = 78
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(a_row_start, a_row_end, b_col_start, b_col_end, c_row_start, c_col_start, start_col):
        nonlocal U, V, W, col
        # A is (a_row_end - a_row_start) x m
        # B is m x (b_col_end - b_col_start)
        # C is (a_row_end - a_row_start) x (b_col_end - b_col_start)
        a_rows = range(a_row_start, a_row_end)
        b_cols = range(b_col_start, b_col_end)
        
        for i in a_rows:
            for k in b_cols:
                for j in range(m):
                    U[i*m + j, start_col] = 1.0
                    V[j*p + k, start_col] = 1.0
                    W[k*n + i, start_col] = 1.0
                    start_col += 1
        col = start_col
        return start_col

    # Helper to add a Strassen block for 2x2x2
    # U, V, W for 2x2x2 Strassen
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [0, 1, 0, 0, 0, 1, 0],  
        [1, 1, 0, 1, 0, 0, -1]  
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0], 
        [0, 0, 1, 0, 0, 1, 0],  
        [0, 0, 0, 1, 0, 0, 1],  
        [1, 0, -1, 0, 1, 0, 1]  
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 0, -1, 0, 1], 
        [0, 0, 1, 0, -1, 0, 0], 
        [0, 1, 0, 1, 0, 0, 0],  
        [1, 0, 1, 1, -1, 1, 1]  
    ], dtype=float)
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
        nonlocal U, V, W
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = S_U[i*2+j, r]
                    if val != 0:
                        U[(a_row+i)*m + (a_col+j), start_col + r] += val
            for i in range(2):
                for j in range(2):
                    val = S_V[i*2+j, r]
                    if val != 0:
                        V[(b_row+i)*p + (b_col+j), start_col + r] += val
            for i in range(2):
                for j in range(2):
                    val = S_W[i*2+j, r]
                    if val != 0:
                        w_idx = (c_col + j) * n + (c_row + i)
                        W[w_idx, start_col + r] += val
        return start_col + 7

    # Block 1: A[0, :] (1x4) * B[:, :] (4x7) -> C[0, :] (1x7)
    # Schoolbook Rank 28
    col = add_schoolbook(0, 1, 0, 7, 0, 0, col)
    
    # Block 2: A[1:3, :] (2x4) * B[:, 0:3] (4x3) -> C[1:3, 0:3] (2x3)
    # Split B into B[:, 0:2] (4x2) and B[:, 2:3] (4x1)
    # Sub-block 2a: A[1:3, :] (2x4) * B[:, 0:2] (4x2) -> C[1:3, 0:2] (2x2)
    # Split A into A[1:3, 0:2] (2x2) and A[1:3, 2:4] (2x2)
    # Split B into B[0:2, 0:2] (2x2) and B[2:4, 0:2] (2x2)
    # C = A1 B1 + A2 B2
    # A1 (2x2) at (1,0), B1 (2x2) at (0,0) -> C1 (2x2) at (1,0)
    col = add_strassen_block(1, 0, 0, 0, 1, 0, col)
    # A2 (2x2) at (1,2), B2 (2x2) at (2,0) -> C2 (2x2) at (1,0)
    # Note: We must add to the SAME C location, so we just add the Strassen blocks
    # But wait, Strassen computes C = A B.
    # Here we want C = A1 B1 + A2 B2.
    # We run Strassen for (A1, B1) and add to C.
    # We run Strassen for (A2, B2) and add to C.
    # The W matrix entries are added, so this works.
    col = add_strassen_block(1, 2, 2, 0, 1, 0, col)
    
    # Sub-block 2b: A[1:3, :] (2x4) * B[:, 2:3] (4x1) -> C[1:3, 2] (2x1)
    # Schoolbook Rank 8 (2 rows * 4 cols * 1 col? No, 2 outputs, each dot product of length 4. Rank 8)
    col = add_schoolbook(1, 3, 2, 3, 1, 2, col)
    
    # Block 3: A[1:3, :] (2x4) * B[:, 3:7] (4x4) -> C[1:3, 3:7] (2x4)
    # Split B into B[:, 3:5] (4x2) and B[:, 5:7] (4x2)
    # Sub-block 3a: A[1:3, :] (2x4) * B[:, 3:5] (4x2) -> C[1:3, 3:5] (2x2)
    # Split A into A[1:3, 0:2] (2x2) and A[1:3, 2:4] (2x2)
    # Split B into B[0:2, 3:5] (2x2) and B[2:4, 3:5] (2x2)
    col = add_strassen_block(1, 0, 0, 3, 1, 3, col)
    col = add_strassen_block(1, 2, 2, 3, 1, 3, col)
    
    # Sub-block 3b: A[1:3, :] (2x4) * B[:, 5:7] (4x2) -> C[1:3, 5:7] (2x2)
    col = add_strassen_block(1, 0, 0, 5, 1, 5, col)
    col = add_strassen_block(1, 2, 2, 5, 1, 5, col)
    
    # Verify
    U_np = U
    V_np = V
    W_np = W
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
