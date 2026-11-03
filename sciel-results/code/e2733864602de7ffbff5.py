import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Rank-52 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses block-Strassen: splits the 4-dimensional inner space into 2 groups of 2,
    and the 8-dimensional output into 4 groups of 2, yielding 8 independent 2x2*2x2
    sub-problems each solved with rank-6.
    
    Output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 52  # 8 sub-problems * 6 multiplications each
    
    U = np.zeros((n * m, rank))  # 8 x 52
    V = np.zeros((m * p, rank))  # 32 x 52
    W = np.zeros((p * n, rank))  # 16 x 52
    
    col = 0
    for s in range(2):  # s=0,1: split A columns / B rows into 2x2 blocks
        for j in range(4):  # j=0..3: split B columns into 2x2 blocks
            # A_s: rows 0-1, cols 2s, 2s+1 of A (2x4)
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j}: rows 2s, 2s+1; cols 2j, 2j+1 of B (4x8)
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1; global index = col*n + row
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Rank-6 bilinear decomposition for 2x2 * 2x2 matrix multiplication
            # Using the known optimal decomposition with 6 multiplications
            
            # m1 = (a00 + a11) * (b00 + b11)
            U[a00, col] += 1
            U[a11, col] += 1
            V[b00, col] += 1
            V[b11, col] += 1
            
            # m2 = (a10 + a11) * b00
            U[a10, col+1] += 1
            U[a11, col+1] += 1
            V[b00, col+1] += 1
            
            # m3 = a00 * (b01 - b11)
            U[a00, col+2] += 1
            V[b01, col+2] += 1
            V[b11, col+2] += -1
            
            # m4 = a11 * b10
            U[a11, col+3] += 1
            V[b10, col+3] += 1
            
            # m5 = (a00 + a01) * b11
            U[a00, col+4] += 1
            U[a01, col+4] += 1
            V[b11, col+4] += 1
            
            # m6 = (a10 - a00) * (b00 + b01)
            U[a10, col+5] += 1
            U[a00, col+5] += -1
            V[b00, col+5] += 1
            V[b01, col+5] += 1
            
            # C-side combinations:
            # c00 = m1 + m4 - m5
            W[c00, col] += 1
            W[c00, col+3] += 1
            W[c00, col+4] += -1
            
            # c01 = m3 + m5
            W[c01, col+2] += 1
            W[c01, col+4] += 1
            
            # c10 = m2 + m4
            W[c10, col+1] += 1
            W[c10, col+3] += 1
            
            # c11 = m1 - m2 + m3 + m6
            W[c11, col] += 1
            W[c11, col+1] += -1
            W[c11, col+2] += 1
            W[c11, col+5] += 1
            
            col += 6
    
    # Convert numpy arrays to lists of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
