import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen terms
    # X is A[i_offset:i_offset+2, j_offset_x:j_offset_x+2]
    # Y is B[j_offset_y:j_offset_y+2, k_offset:k_offset+2]
    # Output adds to C[i_offset:i_offset+2, k_offset:k_offset+2]
    def add_strassen(i_off, jx_off, jy_off, k_off, start_col):
        # Strassen products
        # M1 = (X00 + X11) * (Y00 + Y11)
        # M2 = (X10 + X11) * Y00
        # M3 = X00 * (Y10 - Y11)
        # M4 = X11 * (Y01 - Y00)
        # M5 = (X00 + X01) * Y11
        # M6 = (X10 - X00) * (Y00 + Y01)
        # M7 = (X01 - X11) * (Y10 + Y11)
        
        # Indices in X (A): i in {0,1}, j in {0,1} relative to offsets
        # Indices in Y (B): j in {0,1}, k in {0,1} relative to offsets
        
        # Map local indices to global
        # A index: (i_off + i) * m + (jx_off + j)
        # B index: (jy_off + j) * p + (k_off + k)
        # C index: (k_off + k) * n + (i_off + i)
        
        # M1
        r = start_col
        # U
        U[(i_off + 0) * m + (jx_off + 0), r] += 1
        U[(i_off + 1) * m + (jx_off + 1), r] += 1
        # V
        V[(jy_off + 0) * p + (k_off + 0), r] += 1
        V[(jy_off + 1) * p + (k_off + 1), r] += 1
        # W
        # Z00 = M1 + ... -> W gets +1 for Z00
        # Z01 = ...
        # Z10 = ...
        # Z11 = ...
        # We need to distribute M1 to the appropriate W entries.
        # Z00: i=0, k=0 -> W[k*n+i] = W[0*n+0]
        # Z01: i=0, k=1 -> W[1*n+0]
        # Z10: i=1, k=0 -> W[0*n+1]
        # Z11: i=1, k=1 -> W[1*n+1]
        
        # M1 contributes to Z00 (+1), Z11 (+1)
        W[(k_off + 0) * n + (i_off + 0), r] += 1
        W[(k_off + 1) * n + (i_off + 1), r] += 1
        col = r + 1
        
        # M2 = (X10 + X11) * Y00
        r = start_col + 1
        U[(i_off + 1) * m + (jx_off + 0), r] += 1
        U[(i_off + 1) * m + (jx_off + 1), r] += 1
        V[(jy_off + 0) * p + (k_off + 0), r] += 1
        # M2 contributes to Z10 (+1), Z11 (-1)
        W[(k_off + 0) * n + (i_off + 1), r] += 1
        W[(k_off + 1) * n + (i_off + 1), r] += -1
        col = r + 1
        
        # M3 = X00 * (Y10 - Y11)
        r = start_col + 2
        U[(i_off + 0) * m + (jx_off + 0), r] += 1
        V[(jy_off + 1) * p + (k_off + 0), r] += 1
        V[(jy_off + 1) * p + (k_off + 1), r] += -1
        # M3 contributes to Z01 (+1), Z11 (+1)
        W[(k_off + 1) * n + (i_off + 0), r] += 1
        W[(k_off + 1) * n + (i_off + 1), r] += 1
        col = r + 1
        
        # M4 = X11 * (Y01 - Y00)
        r = start_col + 3
        U[(i_off + 1) * m + (jx_off + 1), r] += 1
        V[(jy_off + 0) * p + (k_off + 1), r] += 1
        V[(jy_off + 0) * p + (k_off + 0), r] += -1
        # M4 contributes to Z00 (+1), Z10 (+1)
        W[(k_off + 0) * n + (i_off + 0), r] += 1
        W[(k_off + 0) * n + (i_off + 1), r] += 1
        col = r + 1
        
        # M5 = (X00 + X01) * Y11
        r = start_col + 4
        U[(i_off + 0) * m + (jx_off + 0), r] += 1
        U[(i_off + 0) * m + (jx_off + 1), r] += 1
        V[(jy_off + 1) * p + (k_off + 1), r] += 1
        # M5 contributes to Z00 (-1), Z01 (+1)
        W[(k_off + 0) * n + (i_off + 0), r] += -1
        W[(k_off + 1) * n + (i_off + 0), r] += 1
        col = r + 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        r = start_col + 5
        U[(i_off + 1) * m + (jx_off + 0), r] += 1
        U[(i_off + 0) * m + (jx_off + 0), r] += -1
        V[(jy_off + 0) * p + (k_off + 0), r] += 1
        V[(jy_off + 0) * p + (k_off + 1), r] += 1
        # M6 contributes to Z11 (+1)
        W[(k_off + 1) * n + (i_off + 1), r] += 1
        col = r + 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        r = start_col + 6
        U[(i_off + 0) * m + (jx_off + 1), r] += 1
        U[(i_off + 1) * m + (jx_off + 1), r] += -1
        V[(jy_off + 1) * p + (k_off + 0), r] += 1
        V[(jy_off + 1) * p + (k_off + 1), r] += 1
        # M7 contributes to Z00 (+1)
        W[(k_off + 0) * n + (i_off + 0), r] += 1
        col = r + 1
        
        return col

    # Part 1: A1 (rows 0,1) * B
    # A11 (cols 0,1) * B11 (rows 0,1, cols 0,1)
    col = add_strassen(0, 0, 0, 0, col)
    # A12 (cols 2,3) * B21 (rows 2,3, cols 0,1)
    col = add_strassen(0, 2, 2, 0, col)
    # A11 (cols 0,1) * B12 (rows 0,1, cols 2,3)
    col = add_strassen(0, 0, 0, 2, col)
    # A12 (cols 2,3) * B22 (rows 2,3, cols 2,3)
    col = add_strassen(0, 2, 2, 2, col)
    
    # Part 2: A2 (row 2) * B
    # Naive: 4 * 4 = 16 terms
    for j in range(4):
        for k in range(4):
            r = col
            U[2 * m + j, r] = 1
            V[j * p + k, r] = 1
            W[k * n + 2, r] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
