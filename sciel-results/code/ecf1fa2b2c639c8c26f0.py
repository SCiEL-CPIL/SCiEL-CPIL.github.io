"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with convention W[k*n+i, r]
    
    The tensor entry T[i, j, k] (where C[i,k] = sum_j A[i,j]B[j,k]) is 1 if (i,j,k) is a valid index.
    The decomposition says:
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    We check this for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split B (2x5) into B1 (2x2), B2 (2x2), B3 (2x1)
    # Use Strassen for B1 and B2 (7 each), rank-1 for B3
    # Total rank: 7 + 7 + 1 = 15
    
    rank = 15
    U = [[0]*rank for _ in range(n*m)]  # 4 x 18
    V = [[0]*rank for _ in range(m*p)]  # 10 x 18
    W = [[0]*rank for _ in range(p*n)]  # 10 x 18
    
    # Strassen helper: adds 7 columns for 2x2 by 2x2 block
    # col_offset: starting column index
    # b_col_offset: offset for B column indices (0, 2, or 4 for B1, B2, B3)
    # c_col_offset: offset for C column indices (0, 2, or 4)
    def add_strassen(U, V, W, col_offset, b_col_offset, c_col_offset):
        # P1 = A00 * (B00 + B11)
        U[0][col_offset] = 1
        V[0 + b_col_offset][col_offset] = 1
        V[3 + b_col_offset][col_offset] = 1
        W[0 + c_col_offset][col_offset] = 1
        W[3 + c_col_offset][col_offset] = 1
        
        # P2 = (A10 + A11) * B00
        U[2][col_offset+1] = 1
        U[3][col_offset+1] = 1
        V[0 + b_col_offset][col_offset+1] = 1
        W[2 + c_col_offset][col_offset+1] = 1
        
        # P3 = (A00 + A01) * B11
        U[0][col_offset+2] = 1
        U[1][col_offset+2] = 1
        V[3 + b_col_offset][col_offset+2] = 1
        W[1 + c_col_offset][col_offset+2] = 1
        W[3 + c_col_offset][col_offset+2] = 1
        
        # P4 = A11 * (B01 - B00)
        U[3][col_offset+3] = 1
        V[1 + b_col_offset][col_offset+3] = 1
        V[0 + b_col_offset][col_offset+3] = -1
        W[0 + c_col_offset][col_offset+3] = 1
        
        # P5 = A00 * (B10 - B11)
        U[0][col_offset+4] = 1
        V[2 + b_col_offset][col_offset+4] = 1
        V[3 + b_col_offset][col_offset+4] = -1
        W[0 + c_col_offset][col_offset+4] = -1
        W[2 + c_col_offset][col_offset+4] = 1
        
        # P6 = (A10 - A00) * B00
        U[2][col_offset+5] = 1
        U[0][col_offset+5] = -1
        V[0 + b_col_offset][col_offset+5] = 1
        W[3 + c_col_offset][col_offset+5] = 1
        
        # P7 = (A01 - A11) * B11
        U[1][col_offset+6] = 1
        U[3][col_offset+6] = -1
        V[3 + b_col_offset][col_offset+6] = 1
        W[0 + c_col_offset][col_offset+6] = -1
        W[1 + c_col_offset][col_offset+6] = 1
    
    # Add Strassen for B1 (cols 0,1 of B -> cols 0,1 of C)
    add_strassen(U, V, W, 0, 0, 0)
    
    # Add Strassen for B2 (cols 2,3 of B -> cols 2,3 of C)
    add_strassen(U, V, W, 7, 2, 2)
    
    # Rank-1 decomposition for B3 (col 4 of B -> col 4 of C)
    # T[i,j,4] = 1 for all i,j
    # U[i*m+j, 14] = 1, V[j*p+4, 14] = 1, W[4*n+i, 14] = 1
    U[0][14] = 1  # i=0,j=0
    U[1][14] = 1  # i=0,j=1
    U[2][14] = 1  # i=1,j=0
    U[3][14] = 1  # i=1,j=1
    V[0 + 4][14] = 1  # j=0, k=4 -> B[0,4]
    V[1 + 4][14] = 1  # j=1, k=4 -> B[1,4]
    W[4 * 2 + 0][14] = 1  # k=4, i=0 -> C[0,4]
    W[4 * 2 + 1][14] = 1  # k=4, i=1 -> C[1,4]
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook
    rank_fb = n * m * p
    U = [[0] * rank_fb for _ in range(n * m)]
    V = [[0] * rank_fb for _ in range(m * p)]
    W = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
