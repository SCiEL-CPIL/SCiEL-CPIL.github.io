<<<<<<< SEARCH
def add_strassen_block(U, V, W, a_cols, b_rows, c_cols, base_col):
=======
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    col_next = add_strassen_block(U, V, W, [0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    col_next = add_strassen_block(U, V, W, [2, 3], [2, 3], [0, 1], col_next)
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    col_next = add_strassen_block(U, V, W, [0, 1], [0, 1], [2, 3], col_next)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    col_next = add_strassen_block(U, V, W, [2, 3], [2, 3], [2, 3], col_next)
    
    return {'U': U, 'V': V, 'W': W}

def add_strassen_block(U, V, W, a_cols, b_rows, c_cols, base_col):
>>>>>>> REPLACE
