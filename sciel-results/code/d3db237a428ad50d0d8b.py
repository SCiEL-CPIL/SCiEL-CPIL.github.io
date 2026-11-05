import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We know rank 28 works. This is the minimal known rank for M_{2,4,4}.
    # The tensor T is in R^{8 x 16 x 8}.
    # We look for rank-r decomposition: T = sum_{k=1}^r u_k \otimes v_k \otimes w_k
    # where u_k in R^8, v_k in R^16, w_k in R^8.
    # Entries are integers.
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen's 7 terms for 2x2 matrix multiplication
        # X_rows: indices in U corresponding to the 2x2 block of A
        #          [A00, A01, A10, A11]
        # Y_rows: indices in V corresponding to the 2x2 block of B
        #          [B00, B01, B10, B11]
        # Z_rows: indices in W corresponding to the 2x2 block of C
        #          [C00, C01, C10, C11]
        
        # M1: (A00 + A11) * (B00 + B11) -> C00, C11
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2: (A10 + A11) * B00 -> C10, -C11
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3: A00 * (B01 - B11) -> C01, C11
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4: A11 * (B10 - B00) -> C00, C10
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5: (A00 + A01) * B11 -> -C00, C01
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6: (A10 - A00) * (B00 + B01) -> C11
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7: (A01 - A11) * (B10 + B11) -> C00
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # A_L indices in U:
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    X1 = [0, 1, 4, 5]
    # B_TL indices in V:
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    Y1 = [0, 1, 4, 5]
    # C_L indices in W:
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # Note: W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 1
    # C[1,0] -> k=0, i=1 -> 2
    # C[1,1] -> k=1, i=1 -> 3
    # So Z1 should map [C00, C01, C10, C11] to [0, 1, 2, 3]
    # Wait, the add_block function expects Z_rows[0]=C00, Z_rows[1]=C01, Z_rows[2]=C10, Z_rows[3]=C11.
    # So Z1 = [0, 1, 2, 3].
    # Let's re-verify the W indexing in the task description: "W[k*n+i,r]".
    # C is n x p. C[i,k].
    # W index for C[i,k] is k*n + i.
    # C_L is the left half of C, so k=0,1.
    # C_L[0,0] is C[0,0] -> idx 0*2+0=0.
    # C_L[0,1] is C[0,1] -> idx 1*2+0=2.
    # C_L[1,0] is C[1,0] -> idx 0*2+1=1.
    # C_L[1,1] is C[1,1] -> idx 1*2+1=3.
    # So the mapping from local (row, col) of C_L to W index:
    # (0,0) -> 0
    # (0,1) -> 2
    # (1,0) -> 1
    # (1,1) -> 3
    # So Z1 = [0, 2, 1, 3]. This matches the current program.
    
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # A_R indices in U:
    # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    X2 = [2, 3, 6, 7]
    # B_BL indices in V:
    # B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    Y2 = [8, 9, 12, 13]
    # C_L indices in W: same as Block 1
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # A_L indices in U: same as Block 1
    X3 = [0, 1, 4, 5]
    # B_TR indices in V:
    # B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    Y3 = [2, 3, 6, 7]
    # C_R indices in W:
    # C_R is the right half of C, so k=2,3.
    # C_R[0,0] is C[0,2] -> idx 2*2+0=4.
    # C_R[0,1] is C[0,3] -> idx 3*2+0=6.
    # C_R[1,0] is C[1,2] -> idx 2*2+1=5.
    # C_R[1,1] is C[1,3] -> idx 3*2+1=7.
    # So Z3 = [4, 6, 5, 7].
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    # A_R indices in U: same as Block 2
    X4 = [2, 3, 6, 7]
    # B_BR indices in V:
    # B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    Y4 = [10, 11, 14, 15]
    # C_R indices in W: same as Block 3
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}
