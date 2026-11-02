import time
import numpy as np

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U row: i*2 + j (i,j in 0..1)
    V row: j*2 + k (j,k in 0..1)
    W row: k*2 + i (k,i in 0..1)
    
    C[i,k] = sum_j A[i,j] B[j,k]
    W[k*2+i, r] gives coefficient for C[i,k] from product r
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrices:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U rows: i*2+j (A[i,j])
    # V rows: j*2+k (B[j,k])
    # W rows: k*2+i (C[i,k])
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00 (row 0) + A11 (row 3)
    # B part: B00 (row 0) + B11 (row 3)
    # C part: C00 (W row 0) + C11 (W row 3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A part: A10 (row 2) + A11 (row 3)
    # B part: B00 (row 0)
    # C part: C10 (W row 1) + C00 (W row 0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[0, 1] = 1
    
    # M3 = A00(B01-B11)
    # A part: A00 (row 0)
    # B part: B01 (row 1) - B11 (row 3)
    # C part: C01 (W row 2) + C11 (W row 3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    # A part: A11 (row 3)
    # B part: B10 (row 2) - B00 (row 0)
    # C part: C00 (W row 0) + C10 (W row 1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    # A part: A00 (row 0) + A01 (row 1)
    # B part: B11 (row 3)
    # C part: C01 (W row 2) - C00 (W row 0)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    # A part: A10 (row 2) - A00 (row 0)
    # B part: B00 (row 0) + B01 (row 1)
    # C part: C11 (W row 3)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A part: A01 (row 1) - A11 (row 3)
    # B part: B10 (row 2) + B11 (row 3)
    # C part: C00 (W row 0) + C11 (W row 3)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    
    4x4 matrices are viewed as 2x2 blocks of 2x2 submatrices.
    Block layout: 00, 01, 10, 11
    Block 00: rows 0-3, Block 01: rows 4-7, Block 10: rows 8-11, Block 11: rows 12-15
    """
    U2, V2, W2 = _strassen_2x2_factors()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    block_offsets = {0: 0, 1: 4, 2: 8, 3: 12}
    
    # 7 Strassen products at the block level:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00       -> contributes to C10, C00
    # M3 = A00(B01-B11)       -> contributes to C01, C11
    # M4 = A11(B10-B00)       -> contributes to C00, C10
    # M5 = (A00+A01)B11       -> contributes to C01, C00 (with -1 on C00)
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00, C11
    
    products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (0, 1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),          # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])   # M7
    ]
    
    for r in range(7):
        A_blocks, B_blocks, C_blocks = products[r]
        
        # Add U contributions: U4[i*4+j, 7*block_r + r]
        for i2 in range(2):
            for j2 in range(2):
                val_2x2 = U2[i2 * 2 + j2, r]
                if val_2x2 == 0:
                    continue
                for block_id, sign in A_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + i2 * 2 + j2
                    U4[row_4x4, 7 * block_id + r] += val_2x2 * sign
                    
        # Add V contributions: V4[j*4+k, 7*block_r + r]
        for j2 in range(2):
            for k2 in range(2):
                val_2x2 = V2[j2 * 2 + k2, r]
                if val_2x2 == 0:
                    continue
                for block_id, sign in B_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + j2 * 2 + k2
                    V4[row_4x4, 7 * block_id + r] += val_2x2 * sign
                    
        # Add W contributions: W4[k*4+i, 7*block_r + r]
        for k2 in range(2):
            for i2 in range(2):
                val_2x2 = W2[k2 * 2 + i2, r]
                if val_2x2 == 0:
                    continue
                for block_id, sign in C_blocks:
                    offset = block_offsets[block_id]
                    row_4x4 = offset + k2 * 2 + i2
                    W4[row_4x4, 7 * block_id + r] += val_2x2 * sign
                    
    return U4, V4, W4

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    
    # Start with 4x4x4 Strassen (rank 49)
    U4, V4, W4 = _strassen_4x4_factors()
    
    # Extend to 4x4x8 by viewing the 8 columns as two blocks of 4
    # The 4x4x8 tensor can be decomposed using the 4x4x4 decomposition
    # applied to two independent 4-column blocks.
    # 
    # U: shape (n*m, rank) = (16, 98)
    #   U[i*m+j, r] for r in 0..47: first 4x4x4 block
    #   U[i*m+j, 49+r] for r in 0..47: second 4x4x4 block (same as first)
    #
    # V: shape (m*p, rank) = (32, 98)
    #   V[j*p+k, r] for k in 0..3: first block, same as V4[j*4+k, r]
    #   V[j*p+k, 49+r] for k in 4..7: second block, same as V4[j*4+(k-4), r]
    #
    # W: shape (p*n, rank) = (32, 98)
    #   W[k*n+i, r] for k in 0..3: first block, same as W4[k*4+i, r]
    #   W[k*n+i, 49+r] for k in 4..7: second block, same as W4[(k-4)*4+i, r]
    
    rank = 98
    U_full = [[0] * rank for _ in range(16)]
    V_full = [[0] * rank for _ in range(32)]
    W_full = [[0] * rank for _ in range(32)]
    
    # Fill first 49 columns
    for i in range(16):
        for r in range(49):
            U_full[i][r] = int(U4[i, r])
    
    for j in range(4):
        for k in range(4):
            v_row = j * 8 + k
            for r in range(49):
                V_full[v_row][r] = int(V4[j * 4 + k, r])
    
    for k in range(4):
        for i in range(4):
            w_row = k * 4 + i
            for r in range(49):
                W_full[w_row][r] = int(W4[k * 4 + i, r])
    
    # Fill second 49 columns (same pattern, shifted)
    for i in range(16):
        for r in range(49):
            U_full[i][49 + r] = int(U4[i, r])
    
    for j in range(4):
        for k in range(4):
            v_row = j * 8 + (4 + k)
            for r in range(49):
                V_full[v_row][49 + r] = int(V4[j * 4 + k, r])
    
    for k in range(4):
        for i in range(4):
            w_row = (4 + k) * 4 + i
            for r in range(49):
                W_full[w_row][49 + r] = int(W4[k * 4 + i, r])
    
    # Validate
    if _validate(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
