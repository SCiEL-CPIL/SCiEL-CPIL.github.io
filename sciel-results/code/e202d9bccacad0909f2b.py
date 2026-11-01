import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1: A11+A22, B11+B22 -> C11(+), C22(+)
    # M2: A21+A22, B11 -> C21(+)
    # M3: A11, B12-B22 -> C12(+)
    # M4: A22, B21-B11 -> C11(+), C21(+)
    # M5: A11+A12, B22 -> C11(-), C12(+)
    # M6: A21-A11, B11+B12 -> C22(+)
    # M7: A12-A22, B21+B22 -> C11(+)
    
    def add_strassen_block(offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # a_row_start, a_col_start: top-left of A block (2x2)
        # b_row_start, b_col_start: top-left of B block (2x2)
        # c_row_start, c_col_start: top-left of C block (2x2)
        
        # A indices: (a_row_start, a_col_start), (a_row_start, a_col_start+1), 
        #            (a_row_start+1, a_col_start), (a_row_start+1, a_col_start+1)
        # U index: i*m + j
        
        # B indices: (b_row_start, b_col_start), ...
        # V index: j*p + k
        
        # C indices: (c_row_start, c_col_start), ...
        # W index: k*n + i
        
        # M1
        U[a_row_start * m + a_col_start][offset] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][offset] += 1
        V[b_row_start * p + b_col_start][offset] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][offset] += 1
        W[c_col_start * n + c_row_start][offset] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][offset] += 1
        
        # M2
        U[(a_row_start+1) * m + a_col_start][offset+1] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][offset+1] += 1
        V[b_row_start * p + b_col_start][offset+1] += 1
        W[(c_col_start) * n + (c_row_start+1)][offset+1] += 1
        
        # M3
        U[a_row_start * m + a_col_start][offset+2] += 1
        V[b_row_start * p + (b_col_start+1)][offset+2] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][offset+2] -= 1
        W[(c_col_start+1) * n + c_row_start][offset+2] += 1
        
        # M4
        U[(a_row_start+1) * m + (a_col_start+1)][offset+3] += 1
        V[(b_row_start+1) * p + b_col_start][offset+3] += 1
        V[b_row_start * p + b_col_start][offset+3] -= 1
        W[c_col_start * n + c_row_start][offset+3] += 1
        W[(c_col_start) * n + (c_row_start+1)][offset+3] += 1
        
        # M5
        U[a_row_start * m + a_col_start][offset+4] += 1
        U[a_row_start * m + (a_col_start+1)][offset+4] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][offset+4] += 1
        W[c_col_start * n + c_row_start][offset+4] -= 1
        W[(c_col_start+1) * n + c_row_start][offset+4] += 1
        
        # M6
        U[(a_row_start+1) * m + a_col_start][offset+5] += 1
        U[a_row_start * m + a_col_start][offset+5] -= 1
        V[b_row_start * p + b_col_start][offset+5] += 1
        V[b_row_start * p + (b_col_start+1)][offset+5] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][offset+5] += 1
        
        # M7
        U[a_row_start * m + (a_col_start+1)][offset+6] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][offset+6] -= 1
        V[(b_row_start+1) * p + b_col_start][offset+6] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][offset+6] += 1
        W[c_col_start * n + c_row_start][offset+6] += 1

    # Block 1: A_L B_TL -> C_TL
    # A_L: rows 0,1 cols 0,1. B_TL: rows 0,1 cols 0,1. C_TL: rows 0,1 cols 0,1.
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_R B_BL -> C_BL (which is C[:, 0:2])
    # A_R: rows 0,1 cols 2,3. B_BL: rows 2,3 cols 0,1. C_BL: rows 0,1 cols 0,1.
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_L B_TR -> C_TR (which is C[:, 2:4])
    # A_L: rows 0,1 cols 0,1. B_TR: rows 0,1 cols 2,3. C_TR: rows 0,1 cols 2,3.
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_R B_BR -> C_BR (which is C[:, 2:4])
    # A_R: rows 0,1 cols 2,3. B_BR: rows 2,3 cols 2,3. C_BR: rows 0,1 cols 2,3.
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    return {"U": U, "V": V, "W": W}
