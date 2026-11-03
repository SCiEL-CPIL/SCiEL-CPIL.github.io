import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen blocks (rank 7 each) for total rank 56.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for s in range(2):
        for j in range(4):
            # Indices for A (2x4)
            # s=0: cols 0,1; s=1: cols 2,3
            a_r0_c0 = 0 * m + (2 * s)
            a_r0_c1 = 0 * m + (2 * s + 1)
            a_r1_c0 = 1 * m + (2 * s)
            a_r1_c1 = 1 * m + (2 * s + 1)
            
            # Indices for B (4x8)
            # s=0: rows 0,1; s=1: rows 2,3
            # j=0: cols 0,1; j=1: cols 2,3; etc
            b_r0_c0 = (2 * s) * p + (2 * j)
            b_r0_c1 = (2 * s) * p + (2 * j + 1)
            b_r1_c0 = (2 * s + 1) * p + (2 * j)
            b_r1_c1 = (2 * s + 1) * p + (2 * j + 1)
            
            # Indices for C (2x8)
            # W[k*n+i, r]
            c_r0_c0 = (2 * j) * n + 0
            c_r0_c1 = (2 * j + 1) * n + 0
            c_r1_c0 = (2 * j) * n + 1
            c_r1_c1 = (2 * j + 1) * n + 1
            
            # Strassen 2x2x2 terms
            # m1 = (a00+a11)(b00+b11)
            U[a_r0_c0, col] += 1
            U[a_r1_c1, col] += 1
            V[b_r0_c0, col] += 1
            V[b_r1_c1, col] += 1
            W[c_r0_c0, col] += 1
            W[c_r1_c1, col] += 1
            
            # m2 = (a10+a11)b00
            U[a_r1_c0, col+1] += 1
            U[a_r1_c1, col+1] += 1
            V[b_r0_c0, col+1] += 1
            W[c_r1_c0, col+1] += 1
            W[c_r1_c1, col+1] += -1
            
            # m3 = a00(b01-b11)
            U[a_r0_c0, col+2] += 1
            V[b_r0_c1, col+2] += 1
            V[b_r1_c1, col+2] += -1
            W[c_r0_c1, col+2] += 1
            W[c_r1_c1, col+2] += 1
            
            # m4 = a11*b10
            U[a_r1_c1, col+3] += 1
            V[b_r1_c0, col+3] += 1
            W[c_r0_c0, col+3] += 1
            W[c_r1_c0, col+3] += 1
            
            # m5 = (a00+a01)b11
            U[a_r0_c0, col+4] += 1
            U[a_r0_c1, col+4] += 1
            V[b_r1_c1, col+4] += 1
            W[c_r0_c0, col+4] += -1
            W[c_r0_c1, col+4] += 1
            
            # m6 = (a10-a00)(b00+b01)
            U[a_r1_c0, col+5] += 1
            U[a_r0_c0, col+5] += -1
            V[b_r0_c0, col+5] += 1
            V[b_r0_c1, col+5] += 1
            W[c_r1_c1, col+5] += 1
            
            # m7 = (a01-a11)(b10+b11)
            U[a_r0_c1, col+6] += 1
            U[a_r1_c1, col+6] += -1
            V[b_r1_c0, col+6] += 1
            V[b_r1_c1, col+6] += 1
            W[c_r0_c0, col+6] += 1
            W[c_r1_c1, col+6] += -1
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
