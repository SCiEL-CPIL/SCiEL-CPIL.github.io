import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 108
    # 8 * 7 (4x4x4 via Strassen) + 4 * 7 (4x4x2 via Strassen) + 24 (4x1x6 trivial)
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen basis for 2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Indices in 2x2 block:
    # A: 0=a, 1=b, 2=c, 3=d (row-major: 00, 01, 10, 11)
    # B: 0=e, 1=f, 2=g, 3=h
    
    strassen_A = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1], # P6
        [0, 1, 0, -1]  # P7
    ]
    strassen_B = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    strassen_W = [
        [1, 0, 0, 0, 1, 0, -1, 1], # X11 = P1 + P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0, 0], # X12 = P3 + P5
        [0, 1, 0, 1, 0, 0, 0, 0], # X21 = P2 + P4
        [1, -1, 1, 0, 0, 1, 0, 0]  # X22 = P1 - P2 + P3 + P6
    ]
    # Note: strassen_W is 4x8, mapping 4 outputs to 7 products.
    # W indices in 2x2 output block: 0=X11, 1=X12, 2=X21, 3=X22.

    def add_strassen_block(a_row, a_col, b_row, b_col, w_row, w_col):
        nonlocal col
        for k in range(7):
            # A indices
            for ai in range(4):
                if strassen_A[k][ai] != 0:
                    # A index: (a_row + ai//2) * m + (a_col + ai%2)
                    U[(a_row + ai // 2) * m + (a_col + ai % 2), col] += strassen_A[k][ai]
            
            # B indices
            for bi in range(4):
                if strassen_B[k][bi] != 0:
                    # B index: (b_row + bi//2) * p + (b_col + bi%2)
                    V[(b_row + bi // 2) * p + (b_col + bi % 2), col] += strassen_B[k][bi]
            
            # W indices
            for wi in range(4):
                if strassen_W[wi][k] != 0:
                    # W index: (w_col + wi%2) * n + (w_row + wi//2)
                    # Wait, W is (p*n, rank). Index is k*n + i.
                    # k is col, i is row.
                    # Output block is 2x2 at w_row, w_col.
                    # Output indices:
                    # (w_row, w_col) -> w_col*n + w_row
                    # (w_row, w_col+1) -> (w_col+1)*n + w_row
                    # (w_row+1, w_col) -> w_col*n + (w_row+1)
                    # (w_row+1, w_col+1) -> (w_col+1)*n + (w_row+1)
                    # wi: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    out_r = w_row + wi // 2
                    out_c = w_col + wi % 2
                    W[out_c * n + out_r, col] += strassen_W[wi][k]
            
            col += 1

    # 1. 4x4x4 part (Top-left 4x4 of A, Top-left 4x4 of B)
    # A rows 0-3, cols 0-3. B rows 0-3, cols 0-3.
    # 4 blocks of 2x2.
    # Block (0,0): A[0:2, 0:2], B[0:2, 0:2] -> W[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Block (0,1): A[0:2, 2:4], B[0:2, 0:2]? No.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # We need A11*B11, A11*B12, A12*B21, A12*B22, etc.
    # The 8 products are:
    # 1. A11 * B11
    # 2. A11 * B12
    # 3. A12 * B21
    # 4. A12 * B22
    # 5. A21 * B11
    # 6. A21 * B12
    # 7. A22 * B21
    # 8. A22 * B22
    
    # A11: r0-1, c0-1. B11: r0-1, c0-1.
    # A11: r0-1, c0-1. B12: r0-1, c2-3.
    # A12: r0-1, c2-3. B21: r2-3, c0-1.
    # A12: r0-1, c2-3. B22: r2-3, c2-3.
    # A21: r2-3, c0-1. B11: r0-1, c0-1.
    # A21: r2-3, c0-1. B12: r0-1, c2-3.
    # A22: r2-3, c2-3. B21: r2-3, c0-1.
    # A22: r2-3, c2-3. B22: r2-3, c2-3.
    
    add_strassen_block(0, 0, 0, 2, 0, 0) # A11, B12 -> W11? No, W12 is col 2-3.
    # Correction:
    # A11 * B12: Output is at W rows 0-1, cols 2-3.
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_strassen_block(0, 2, 2, 0, 0, 0) # A12, B21 -> W rows 0-1, cols 0-1.
    add_strassen_block(0, 2, 2, 2, 0, 2) # A12, B22 -> W rows 0-1, cols 2-3.
    add_strassen_block(2, 0, 0, 0, 2, 0) # A21, B11 -> W rows 2-3, cols 0-1.
    add_strassen_block(2, 0, 0, 2, 2, 2) # A21, B12 -> W rows 2-3, cols 2-3.
    add_strassen_block(2, 2, 2, 0, 2, 0) # A22, B21 -> W rows 2-3, cols 0-1.
    add_strassen_block(2, 2, 2, 2, 2, 2) # A22, B22 -> W rows 2-3, cols 2-3.
    
    # 2. 4x4x2 part (Top-left 4x4 of A, Cols 4-5 of B)
    # B rows 0-3, cols 4-5.
    # Split B into B_top (rows 0-1) and B_bot (rows 2-3).
    # A11 * B_top
    add_strassen_block(0, 0, 0, 4, 0, 4)
    # A12 * B_bot
    add_strassen_block(0, 2, 2, 4, 0, 4)
    # A21 * B_top
    add_strassen_block(2, 0, 0, 4, 2, 4)
    # A22 * B_bot
    add_strassen_block(2, 2, 2, 4, 2, 4)
    
    # 3. 4x1x6 part (A col 4, B row 4)
    # A rows 0-3, col 4. B row 4, cols 0-5.
    for i in range(4):
        for j in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + j, col] = 1
            W[j * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
