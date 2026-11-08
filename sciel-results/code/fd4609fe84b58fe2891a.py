import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # 1. A1 (2x2) * B1 (2x6) using 3x Strassen (rank 7 each) -> 21
    # 2. a2 (2x1) * b2 (1x6) using schoolbook -> 12
    # Total 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    col = 0
    
    # Part 1: A1 * B1
    # A1 is A[0:2, 0:2] -> indices 0,1 in U (row 0, 1)
    # B1 is B[0:2, :] -> indices 0:12 in V (rows 0:12)
    # Output C[0:2, 0:6] -> indices 0:12 in W (rows 0:12)
    
    # Strassen for 2x2x2
    # M1 = A00 * B00
    # M2 = A10 * B10
    # M3 = A01 * B01
    # M4 = A11 * B11
    # M5 = (A00+A11) * (B00+B11)
    # M6 = (A10+A11) * (B01-B00)
    # M7 = (A00+A01) * (B10-B11)
    # C00 = M5 + M4 - M2 - M7
    # C01 = M1 + M7
    # C10 = M3 + M6
    # C11 = M5 + M1 - M4 - M6
    
    # We have 3 blocks of B1: B1_0 (cols 0,1), B1_1 (cols 2,3), B1_2 (cols 4,5)
    for block in range(3):
        # B1_block is 2x2, rows 0,1 of B, cols 2*block, 2*block+1
        # V indices for B1_block:
        # B[0, 2*block] -> V[0 + 2*block]
        # B[0, 2*block+1] -> V[0 + 2*block+1]
        # B[1, 2*block] -> V[3 + 2*block]
        # B[1, 2*block+1] -> V[3 + 2*block+1]
        
        # W indices for C block:
        # C[0, 2*block] -> W[0 + 2*block]
        # C[0, 2*block+1] -> W[0 + 2*block+1]
        # C[1, 2*block] -> W[6 + 2*block]
        # C[1, 2*block+1] -> W[6 + 2*block+1]
        
        # U indices for A1:
        # A[0,0] -> U[0]
        # A[0,1] -> U[1]
        # A[1,0] -> U[3]
        # A[1,1] -> U[4]
        
        # Strassen terms
        # M1 = A00 * B00
        U[0][col] = 1
        V[2*block][col] = 1
        # W contribution: C00 += M1, C01 += M1? No.
        # C00 = M5 + M4 - M2 - M7
        # C01 = M1 + M7
        # C10 = M3 + M6
        # C11 = M5 + M1 - M4 - M6
        
        # Let's map the 7 Strassen multiplies to columns col..col+6
        
        # 1. M1 = A00 * B00
        U[0][col] = 1
        V[2*block][col] = 1
        W[1 + 2*block][col] = 1 # C01
        W[6 + 1 + 2*block][col] = 1 # C11
        col += 1
        
        # 2. M2 = A10 * B10
        U[3][col] = 1
        V[3 + 2*block][col] = 1
        W[0 + 2*block][col] = -1 # C00
        W[6 + 0 + 2*block][col] = 0 # C10? No, M2 is not in C10
        # Wait, C10 = M3 + M6. M2 is not there.
        # C11 = M5 + M1 - M4 - M6. M2 is not there.
        # So M2 only affects C00.
        W[0 + 2*block][col] = -1
        col += 1
        
        # 3. M3 = A01 * B01
        U[1][col] = 1
        V[2*block+1][col] = 1
        W[6 + 0 + 2*block][col] = 1 # C10
        col += 1
        
        # 4. M4 = A11 * B11
        U[4][col] = 1
        V[3 + 2*block+1][col] = 1
        W[0 + 2*block][col] = 1 # C00
        W[6 + 1 + 2*block][col] = -1 # C11
        col += 1
        
        # 5. M5 = (A00+A11) * (B00+B11)
        U[0][col] = 1
        U[4][col] = 1
        V[2*block][col] = 1
        V[3 + 2*block+1][col] = 1
        W[0 + 2*block][col] = 1 # C00
        W[6 + 1 + 2*block][col] = 1 # C11
        col += 1
        
        # 6. M6 = (A10+A11) * (B01-B00)
        U[3][col] = 1
        U[4][col] = 1
        V[2*block+1][col] = 1
        V[2*block][col] = -1
        W[6 + 0 + 2*block][col] = 1 # C10
        W[6 + 1 + 2*block][col] = -1 # C11
        col += 1
        
        # 7. M7 = (A00+A01) * (B10-B11)
        U[0][col] = 1
        U[1][col] = 1
        V[3 + 2*block][col] = 1
        V[3 + 2*block+1][col] = -1
        W[0 + 2*block][col] = -1 # C00
        W[1 + 2*block][col] = 1 # C01
        col += 1
        
    # Part 2: a2 * b2
    # a2 is A[0:2, 2] -> U[2], U[5]
    # b2 is B[2, :] -> V[12:18]
    # C2 is C[0:2, 0:6] -> W[0:12]
    # C[0, k] += A[0,2] * B[2,k]
    # C[1, k] += A[1,2] * B[2,k]
    
    for k in range(6):
        # Term 1: A[0,2] * B[2,k]
        U[2][col] = 1
        V[12 + k][col] = 1
        W[k][col] = 1 # C[0,k]
        col += 1
        
        # Term 2: A[1,2] * B[2,k]
        U[5][col] = 1
        V[12 + k][col] = 1
        W[6 + k][col] = 1 # C[1,k]
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
