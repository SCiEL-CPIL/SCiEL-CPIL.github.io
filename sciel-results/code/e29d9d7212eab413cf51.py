# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Fallback to known rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def _add_strassen_2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        U[a_row * m + a_col][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[b_row * p + b_col][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        U[(a_row+1) * m + a_col][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[b_row * p + b_col][col] = 1
        W[(c_col + 0) * n + (c_row + 1)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = -1
        col += 1
        
        U[a_row * m + a_col][col] = 1
        V[b_row * p + (b_col+1)][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = -1
        W[(c_col + 1) * n + (c_row + 0)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        U[(a_row+1) * m + (a_col+1)][col] = 1
        V[(b_row+1) * p + b_col][col] = 1
        V[b_row * p + b_col][col] = -1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        W[(c_col + 0) * n + (c_row + 1)][col] = 1
        col += 1
        
        U[a_row * m + a_col][col] = 1
        U[a_row * m + (a_col+1)][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = -1
        W[(c_col + 1) * n + (c_row + 0)][col] = 1
        col += 1
        
        U[(a_row+1) * m + a_col][col] = 1
        U[a_row * m + a_col][col] = -1
        V[b_row * p + b_col][col] = 1
        V[b_row * p + (b_col+1)][col] = 1
        W[(c_col + 1) * n + (c_row + 1)][col] = 1
        col += 1
        
        U[a_row * m + (a_col+1)][col] = 1
        U[(a_row+1) * m + (a_col+1)][col] = -1
        V[(b_row+1) * p + b_col][col] = 1
        V[(b_row+1) * p + (b_col+1)][col] = 1
        W[(c_col + 0) * n + (c_row + 0)][col] = 1
        col += 1
    
    def _add_outer_2x1_1x4(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        for i in range(2):
            for j in range(4):
                U[(a_row + i) * m + a_col][col] = 1
                V[(b_row + 0) * p + (b_col + j)][col] = 1
                W[(c_col + j) * n + (c_row + i)][col] = 1
                col += 1

    for blk in range(2):
        bcs = 4 * blk
        ccs = 4 * blk
        _add_strassen_2x2(0, 0, 0, bcs,     0, ccs)
        _add_strassen_2x2(0, 0, 0, bcs + 2, 0, ccs + 2)
        _add_outer_2x1_1x4(0, 2, 2, bcs, 0, ccs)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
