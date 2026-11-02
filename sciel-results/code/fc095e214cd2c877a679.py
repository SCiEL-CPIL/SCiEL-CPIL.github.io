import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank 40
    rank = 40
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A[0:1, :] (1x4) * B[:, 0:3] (4x3) -> C[0:1, 0:3] (1x3)
    # Schoolbook Rank 12
    for j in range(4):
        for k in range(3):
            U[0*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + 0, col] = 1.0
            col += 1
            
    # Part 2: A[1:3, :] (2x4) * B[:, 3:7] (4x4) -> C[1:3, 3:7] (2x4)
    # This part is decomposed into 4 Strassen blocks (Rank 7 each)
    
    # Strassen U, V, W for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # b (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # c (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # d (1,1)
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0], # e (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # f (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # g (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # h (1,1)
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, -1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 1, 1, -1, 1, 1]  # C11
    ], dtype=float)
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
        nonlocal U, V, W
        # Map local 2x2 indices to global
        # Local A[i,j] -> Global A[a_row+i, a_col+j]
        # Local B[i,j] -> Global B[b_row+i, b_col+j]
        # Local C[i,j] -> Global C[c_row+i, c_col+j]
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = S_U[i*2+j, r]
                    if val != 0:
                        U[(a_row+i)*m + (a_col+j), start_col + r] += val
            # V
            for i in range(2):
                for j in range(2):
                    val = S_V[i*2+j, r]
                    if val != 0:
                        V[(b_row+i)*p + (b_col+j), start_col + r] += val
            # W
            for i in range(2):
                for j in range(2):
                    val = S_W[i*2+j, r]
                    if val != 0:
                        W[(c_row+j)*n + (c_row+i), start_col + r] += val
                        # Wait, W index is k*n + i.
                        # Local C[i,j] corresponds to i (row), j (col).
                        # Global C[c_row+i, c_col+j].
                        # W index is (c_col+j)*n + (c_row+i).
                        
    # Correction on W index in add_strassen_block:
    # W[k*n + i]
    # Local C[i,j]: i is row, j is col.
    # Global Row: c_row + i
    # Global Col: c_col + j
    # W Index: (c_col + j) * n + (c_row + i)
    
    # Re-defining add_strassen_block with correct W index
    def add_strassen_block_correct(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
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
                        # W index: col * n + row
                        w_idx = (c_col + j) * n + (c_row + i)
                        W[w_idx, start_col + r] += val

    # Block 1: A[1:3, 0:2] * B[0:2, 3:5] -> C[1:3, 3:5]
    add_strassen_block_correct(1, 0, 0, 3, 1, 3, col)
    col += 7
    
    # Block 2: A[1:3, 2:4] * B[2:4, 3:5] -> C[1:3, 3:5]
    add_strassen_block_correct(1, 2, 2, 3, 1, 3, col)
    col += 7
    
    # Block 3: A[1:3, 0:2] * B[0:2, 5:7] -> C[1:3, 5:7]
    add_strassen_block_correct(1, 0, 0, 5, 1, 5, col)
    col += 7
    
    # Block 4: A[1:3, 2:4] * B[2:4, 5:7] -> C[1:3, 5:7]
    add_strassen_block_correct(1, 2, 2, 5, 1, 5, col)
    col += 7
    
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
