import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _add_strassen_block(U, V, W, start_col, I, J, K):
    """
    Adds a Strassen block to the factor matrices.
    I, J, K are lists of 2 indices each.
    Returns the next column index.
    """
    i0, i1 = I
    j0, j1 = J
    k0, k1 = K
    
    # M1 = (a00 + a11)(b00 + b11)
    U[i0*m + j0][start_col] = 1
    U[i1*m + j1][start_col] = 1
    V[j0*p + k0][start_col] = 1
    V[j1*p + k1][start_col] = 1
    W[k0*n + i0][start_col] = 1
    W[k1*n + i1][start_col] = 1
    start_col += 1
    
    # M2 = (a10 + a11)b00
    U[i1*m + j0][start_col] = 1
    U[i1*m + j1][start_col] = 1
    V[j0*p + k0][start_col] = 1
    W[k0*n + i1][start_col] = 1
    start_col += 1
    
    # M3 = a00(b01 - b11)
    U[i0*m + j0][start_col] = 1
    V[j0*p + k1][start_col] = 1
    V[j1*p + k1][start_col] = -1
    W[k1*n + i0][start_col] = 1
    start_col += 1
    
    # M4 = a11(b10 - b00)
    U[i1*m + j1][start_col] = 1
    V[j1*p + k0][start_col] = 1
    V[j0*p + k0][start_col] = -1
    W[k0*n + i0][start_col] = 1
    W[k0*n + i1][start_col] = 1
    start_col += 1
    
    # M5 = (a00 + a01)b11
    U[i0*m + j0][start_col] = 1
    U[i0*m + j1][start_col] = 1
    V[j1*p + k1][start_col] = 1
    W[k0*n + i0][start_col] = -1
    W[k1*n + i0][start_col] = 1
    start_col += 1
    
    # M6 = (a10 - a00)(b00 + b01)
    U[i1*m + j0][start_col] = 1
    U[i0*m + j0][start_col] = -1
    V[j0*p + k0][start_col] = 1
    V[j0*p + k1][start_col] = 1
    W[k1*n + i1][start_col] = 1
    start_col += 1
    
    # M7 = (a01 - a11)(b10 + b11)
    U[i0*m + j1][start_col] = 1
    U[i1*m + j1][start_col] = -1
    V[j1*p + k0][start_col] = 1
    V[j1*p + k1][start_col] = 1
    W[k0*n + i0][start_col] = 1
    W[k1*n + i1][start_col] = -1
    start_col += 1
    
    return start_col

def _add_schoolbook_block(U, V, W, start_col, I, J, K):
    """
    Adds a schoolbook block to the factor matrices.
    I, J, K are lists of indices.
    Returns the next column index.
    """
    for i in I:
        for j in J:
            for k in K:
                U[i*m + j][start_col] = 1
                V[j*p + k][start_col] = 1
                W[k*n + i][start_col] = 1
                start_col += 1
    return start_col

def _construct_block_strassen():
    n, m, p = DIMENSIONS
    # Total rank estimate: 6 * 7 + 30 + 12 = 84
    max_rank = 100 
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # 6 Strassen blocks
    # I = {0, 1}, J = {0, 1}, K = {0, 1}
    col = _add_strassen_block(U, V, W, col, [0, 1], [0, 1], [0, 1])
    # I = {0, 1}, J = {0, 1}, K = {2, 3}
    col = _add_strassen_block(U, V, W, col, [0, 1], [0, 1], [2, 3])
    # I = {0, 1}, J = {0, 1}, K = {4, 5}
    col = _add_strassen_block(U, V, W, col, [0, 1], [0, 1], [4, 5])
    # I = {0, 1}, J = {2, 3}, K = {0, 1}
    col = _add_strassen_block(U, V, W, col, [0, 1], [2, 3], [0, 1])
    # I = {0, 1}, J = {2, 3}, K = {2, 3}
    col = _add_strassen_block(U, V, W, col, [0, 1], [2, 3], [2, 3])
    # I = {0, 1}, J = {2, 3}, K = {4, 5}
    col = _add_strassen_block(U, V, W, col, [0, 1], [2, 3], [4, 5])
    
    # Schoolbook Block A: i=2, j in 0..4, k in 0..5
    col = _add_schoolbook_block(U, V, W, col, [2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5])
    
    # Schoolbook Block B: i in 0..1, j=4, k in 0..5
    col = _add_schoolbook_block(U, V, W, col, [0, 1], [4], [0, 1, 2, 3, 4, 5])
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Use the block Strassen decomposition which achieves rank 84
    result = _construct_block_strassen()
    
    # Safety check
    if time.time() - start_time > 100:
         return _construct_schoolbook()
         
    return result
# EVOLVE-BLOCK-END
