import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = 2, 3, 7
    
    # We use a block decomposition: M_{2,3,7} = M_{2,3,3} \oplus M_{2,3,3} \oplus M_{2,3,1}
    # Rank of M_{2,3,3} is 7.
    # Rank of M_{2,3,1} is 6 (schoolbook) or 5? Let's use 6 for safety.
    # Total rank = 7 + 7 + 6 = 20.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the rank 7 decomposition for M_{2,3,3}
    # Indices for A (2x3): 0,1,2,3,4,5
    # Indices for B (3x3): 0..8
    # Indices for C (2x3): 0..5 (k*n+i)
    
    # U_block (6x7)
    U_b = [
        [1, 0, 0, 0, 0, 0, 1], # A00
        [0, 1, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 1], # A02
        [0, 0, 0, 1, 0, 0, 0], # A10
        [0, 0, 0, 0, 1, 0, 1], # A11
        [0, 0, 0, 0, 0, 1, 1]  # A12
    ]
    
    # V_block (9x7)
    V_b = [
        [1, 0, 0, 0, 0, 0, 1], # B00
        [0, 0, 0, 1, 0, 0, -1],# B01
        [0, 0, 0, 0, 0, 0, 0], # B02
        [0, 1, 0, 0, 0, 0, 1], # B10
        [0, 0, 0, 0, 1, 0, 1], # B11
        [0, 0, 0, 0, 0, 0, 0], # B12
        [0, 0, 1, 0, 0, 0, 1], # B20
        [0, 0, 0, 0, 0, 1, 1], # B21
        [0, 0, 0, 0, 0, 0, 0]  # B22
    ]
    
    # W_block (6x7)
    W_b = [
        [1, 0, 0, 0, 0, 0, 1], # C00
        [0, 1, 0, 0, 0, 0, 0], # C01
        [0, 0, 1, 0, 0, 0, 1], # C02
        [0, 0, 0, 1, 0, 0, 0], # C10
        [0, 0, 0, 0, 1, 0, 1], # C11
        [0, 0, 0, 0, 0, 1, 1]  # C12
    ]
    
    # Verify this decomposition?
    # C00 = A00B00 + A01B10 + A02B20
    # P1 = A00B00
    # P2 = A01B10
    # P3 = A02B20
    # P7 = (A00+A11+A02+A12)(B00+B11+B20+B21)
    # W00 = P1 + P7?
    # P1 + P7 = A00B00 + A00B11 + A00B20 + A00B21 + A11B00 + ...
    # This does NOT match C00.
    # The coefficients in W_b are wrong for this U_b/V_b.
    
    # I will fallback to a known correct decomposition or schoolbook.
    # Schoolbook for M_{2,3,3} has rank 9.
    # Schoolbook for M_{2,3,1} has rank 6.
    # Total rank 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: M_{2,3,3} (cols 0-2)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    # Block 2: M_{2,3,3} (cols 3-5)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+(k+3)][col] = 1
                W[(k+3)*2+i][col] = 1
                col += 1
    # Block 3: M_{2,3,1} (col 6)
    for i in range(2):
        for j in range(3):
            U[i*3+j][col] = 1
            V[j*3+6][col] = 1
            W[6*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
