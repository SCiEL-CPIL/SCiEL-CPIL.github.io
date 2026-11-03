"""Rewritten program: rank-55 decomposition for 2x3 x 3x10 matrix multiplication."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0
    SAFETY_MARGIN = 10.0
    
    n, m, p = DIMENSIONS
    
    # Rank 55 decomposition:
    # Split m=3 into 2 (Strassen) + 1 (Naive)
    # Split p=10 into 5 blocks of 2 (Strassen on each block)
    # Strassen per 2x2 block: 7 multiplications
    # 5 blocks * 7 = 35
    # Naive part: 2*1*10 = 20
    # Total: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # (n*m, rank) = (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (m*p, rank) = (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (p*n, rank) = (20, 55)
    
    col = 0
    
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    # A is 2x2 (rows 0,1; cols 0,1), B is 2x10 (rows 0,1; cols 0..9)
    # Split B into 5 blocks of 2 columns each.
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0*3 + 0][col] += 1  # A00
        U[1*3 + 1][col] += 1  # A11
        V[0*10 + k0][col] += 1 # B00
        V[1*10 + k1][col] += 1 # B11
        W[k0*2 + 0][col] += 1  # C00 += M1
        W[k1*2 + 1][col] += 1  # C11 += M1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1*3 + 0][col] += 1  # A10
        U[1*3 + 1][col] += 1  # A11
        V[0*10 + k0][col] += 1 # B00
        W[k0*2 + 1][col] += 1  # C10 += M2
        W[k1*2 + 1][col] += -1 # C11 -= M2
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0*3 + 0][col] += 1  # A00
        V[0*10 + k1][col] += 1 # B01
        V[1*10 + k1][col] += -1 # -B11
        W[k1*2 + 0][col] += 1  # C01 += M3
        W[k1*2 + 1][col] += 1  # C11 += M3
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[1*3 + 1][col] += 1  # A11
        V[1*10 + k0][col] += 1 # B10
        V[0*10 + k0][col] += -1 # -B00
        W[k0*2 + 0][col] += 1  # C00 += M4
        W[k0*2 + 1][col] += 1  # C10 += M4
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0*3 + 0][col] += 1  # A00
        U[0*3 + 1][col] += 1  # A01
        V[1*10 + k1][col] += 1 # B11
        W[k0*2 + 0][col] += -1 # C00 -= M5
        W[k1*2 + 0][col] += 1  # C01 += M5
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1*3 + 0][col] += 1  # A10
        U[0*3 + 0][col] += -1 # -A00
        V[0*10 + k0][col] += 1 # B00
        V[0*10 + k1][col] += 1 # B01
        W[k1*2 + 1][col] += 1  # C11 += M6
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0*3 + 1][col] += 1  # A01
        U[1*3 + 1][col] += -1 # -A11
        V[1*10 + k0][col] += 1 # B10
        V[1*10 + k1][col] += 1 # B11
        W[k0*2 + 0][col] += 1  # C00 += M7
        col += 1

    # Part 2: Naive for A[:, 2] * B[2, :]
    # A is 2x1, B is 1x10
    # C[i,k] += A[i,2] * B[2,k]
    
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] += 1
            V[2*10 + k][col] += 1
            W[k*2 + i][col] += 1
            col += 1
            
    if col != rank:
        # Fallback to schoolbook if construction fails (should not happen)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
