import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    rank = 28
    U = np.zeros((n*m, rank))  # 8 x 28
    V = np.zeros((m*p, rank))  # 16 x 28
    W = np.zeros((p*n, rank))  # 8 x 28
    
    def add_strassen_block(row_offset_X, col_offset_X, row_offset_Y, col_offset_Y, row_offset_Z, col_offset_Z, base_col):
        col = base_col
        
        # M1 = (x11+x22)(y11+y22)
        U[row_offset_X*4 + col_offset_X, col] = 1
        U[(row_offset_X+1)*4 + (col_offset_X+1), col] = 1
        V[row_offset_Y*4 + col_offset_Y, col] = 1
        V[(row_offset_Y+1)*4 + (col_offset_Y+1), col] = 1
        W[col_offset_Z*2 + row_offset_Z, col] = 1
        W[(col_offset_Z+1)*2 + row_offset_Z, col] = 1
        W[col_offset_Z*2 + (row_offset_Z+1), col] = 1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = 1
        col += 1
        
        # M2 = (x21+x22)y11
        U[(row_offset_X+1)*4 + col_offset_X, col] = 1
        U[(row_offset_X+1)*4 + (col_offset_X+1), col] = 1
        V[row_offset_Y*4 + col_offset_Y, col] = 1
        W[col_offset_Z*2 + (row_offset_Z+1), col] = 1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = -1
        col += 1
        
        # M3 = x11(y12-y22)
        U[row_offset_X*4 + col_offset_X, col] = 1
        V[row_offset_Y*4 + (col_offset_Y+1), col] = 1
        V[(row_offset_Y+1)*4 + (col_offset_Y+1), col] = -1
        W[(col_offset_Z+1)*2 + row_offset_Z, col] = 1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = 1
        col += 1
        
        # M4 = x22(y21-y11)
        U[(row_offset_X+1)*4 + (col_offset_X+1), col] = 1
        V[(row_offset_Y+1)*4 + col_offset_Y, col] = 1
        V[row_offset_Y*4 + col_offset_Y, col] = -1
        W[col_offset_Z*2 + row_offset_Z, col] = 1
        W[(col_offset_Z+1)*2 + row_offset_Z, col] = 1
        W[col_offset_Z*2 + (row_offset_Z+1), col] = 1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = 1
        col += 1
        
        # M5 = (x11+x12)y22
        U[row_offset_X*4 + col_offset_X, col] = 1
        U[row_offset_X*4 + (col_offset_X+1), col] = 1
        V[(row_offset_Y+1)*4 + (col_offset_Y+1), col] = 1
        W[col_offset_Z*2 + row_offset_Z, col] = -1
        W[(col_offset_Z+1)*2 + row_offset_Z, col] = 1
        W[col_offset_Z*2 + (row_offset_Z+1), col] = -1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = 1
        col += 1
        
        # M6 = (x21-x11)(y11+y12)
        U[(row_offset_X+1)*4 + col_offset_X, col] = 1
        U[row_offset_X*4 + col_offset_X, col] = -1
        V[row_offset_Y*4 + col_offset_Y, col] = 1
        V[row_offset_Y*4 + (col_offset_Y+1), col] = 1
        W[(col_offset_Z+1)*2 + (row_offset_Z+1), col] = 1
        col += 1
        
        # M7 = (x12-x22)(y21+y22)
        U[row_offset_X*4 + (col_offset_X+1), col] = 1
        U[(row_offset_X+1)*4 + (col_offset_X+1), col] = -1
        V[(row_offset_Y+1)*4 + col_offset_Y, col] = 1
        V[(row_offset_Y+1)*4 + (col_offset_Y+1), col] = 1
        W[col_offset_Z*2 + row_offset_Z, col] = 1
        W[(col_offset_Z+1)*2 + row_offset_Z, col] = 1
        col += 1
        
        return col
    
    # Block 1: A1 * B11 -> C[0:2, 0:2]
    next_col = add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A2 * B21 -> C[0:2, 0:2]
    next_col = add_strassen_block(0, 2, 2, 0, 0, 0, next_col)
    
    # Block 3: A1 * B12 -> C[0:2, 2:4]
    next_col = add_strassen_block(0, 0, 0, 2, 0, 2, next_col)
    
    # Block 4: A2 * B22 -> C[0:2, 2:4]
    next_col = add_strassen_block(0, 2, 2, 2, 0, 2, next_col)
    
    assert next_col == 28
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
