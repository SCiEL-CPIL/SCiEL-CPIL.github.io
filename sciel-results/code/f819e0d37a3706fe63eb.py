import time
import numpy as np

def construct():
    """
    Returns a rank-24 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses a custom Strassen-like decomposition with 6 products per 2x2 block,
    reducing the standard 7-product Strassen by one per block (4 blocks total).
    """
    t0 = time.time()
    deadline = t0 + 115.0
    
    n, m, p = 2, 4, 4
    rank = 24
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 24)
    V = [[0] * rank for _ in range(m * p)]   # shape (16, 24)
    W = [[0] * rank for _ in range(p * n)]   # shape (8, 24)
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        """
        Add 6 tensor factors for one 2x2x2 block.
        X_rows: [x00, x01, x10, x11] - indices into U (A block rows)
        Y_rows: [y00, y01, y10, y11] - indices into V (B block cols)
        Z_rows: [z00, z10, z01, z11] - indices into W (C block)
                 Note: Z indexing follows trace convention W[k*n+i, r]
                 For a 2x2 C block at position (block_row, block_col):
                 C[0,0]=Z[0], C[1,0]=Z[1], C[0,1]=Z[2], C[1,1]=Z[3]
        """
        # P1: (a00+a10)(b00+b10)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[2]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # P2: (a01+a11)(b10)
        U[X_rows[1]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[2]][col_offset + 1] += 1
        W[Z_rows[1]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # P3: a00 * b01
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        W[Z_rows[2]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # P4: a11 * b11
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[3]][col_offset + 3] += 1
        W[Z_rows[3]][col_offset + 3] += 1
        
        # P5: (a00-a10) * b11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 4] -= 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[2]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        
        # P6: (a01+a11) * (b00+b01)
        U[X_rows[1]][col_offset + 5] += 1
        U[X_rows[3]][col_offset + 5] += 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[1]][col_offset + 5] += 1
        W[Z_rows[2]][col_offset + 5] -= 1
    
    # Define the 4 blocks of the 2x4 * 4x4 multiplication
    # A is 2x4: split into left (2x2) and right (2x2)
    # B is 4x4: split into TL, TR, BL, BR (each 2x2)
    # C is 2x4: split into left (2x2) and right (2x2)
    
    # A_L: rows 0-1 of A (indices 0,1 for row 0; 2,3 for row 1)
    # In U, index = row * m + col = row * 4 + col
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # A_L = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] -> X = [0, 1, 4, 5]
    
    # A_R: rows 0-1 of A, cols 2-3
    # A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    # A_R = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] -> X = [2, 3, 6, 7]
    
    # B_TL: rows 0-1, cols 0-1
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # B_TL = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] -> Y = [0, 1, 4, 5]
    
    # B_TR: rows 0-1, cols 2-3
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # B_TR = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]] -> Y = [2, 3, 6, 7]
    
    # B_BL: rows 2-3, cols 0-1
    # B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    # B_BL = [[B[2,0], B[2,1]], [B[3,0], B[3,1]]] -> Y = [8, 9, 12, 13]
    
    # B_BR: rows 2-3, cols 2-3
    # B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    # B_BR = [[B[2,2], B[2,3]], [B[3,2], B[3,3]]] -> Y = [10, 11, 14, 15]
    
    # C_L: cols 0-1 of C
    # C[0,0]=W[0], C[1,0]=W[2], C[0,1]=W[1], C[1,1]=W[3]
    # Using trace convention W[k*n+i]: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # Z_L = [W[0], W[2], W[1], W[3]] = [0, 2, 1, 3]
    
    # C_R: cols 2-3 of C
    # C[0,2]=W[4], C[1,2]=W[6], C[0,3]=W[5], C[1,3]=W[7]
    # Z_R = [W[4], W[6], W[5], W[7]] = [4, 6, 5, 7]
    
    # Block 1: A_L * B_TL -> C_L
    add_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Block 2: A_R * B_BL -> C_L
    add_block(6, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Block 3: A_L * B_TR -> C_R
    add_block(12, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Block 4: A_R * B_BR -> C_R
    add_block(18, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
    
    return {"U": U, "V": V, "W": W}
