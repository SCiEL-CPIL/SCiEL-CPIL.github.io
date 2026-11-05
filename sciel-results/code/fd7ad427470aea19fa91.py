import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 33 decomposition:
    # Split B (3x6) into B_top (2x6) and B_bot (1x6)
    # C = A_top B_top + A_bot B_bot
    # A_top is 2x2, A_bot is 2x1
    # B_top is 2x6 (split into three 2x2 blocks), B_bot is 1x6
    
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 2: A_bot (2x1) * B_bot (1x6)
    # A_bot indices: A[0,2] (U idx 2), A[1,2] (U idx 5)
    # B_bot indices: B[2,k] (V idx 12+k)
    # C indices: C[0,k] (W idx 2k), C[1,k] (W idx 2k+1)
    
    for k in range(6):
        # Term 1: A[0,2] * B[2,k] -> C[0,k]
        U[2, col] = 1.0
        V[12 + k, col] = 1.0
        W[2 * k, col] = 1.0
        col += 1
        
        # Term 2: A[1,2] * B[2,k] -> C[1,k]
        U[5, col] = 1.0
        V[12 + k, col] = 1.0
        W[2 * k + 1, col] = 1.0
        col += 1
    
    # Part 1: A_top (2x2) * B_top (2x6)
    # A_top indices:
    # A[0,0] (U idx 0), A[0,1] (U idx 1)
    # A[1,0] (U idx 3), A[1,1] (U idx 4)
    
    # B_top is split into 3 blocks of 2x2.
    # Block t (0..2) uses B cols 2t, 2t+1.
    # B[0, 2t] (V idx 2t), B[0, 2t+1] (V idx 2t+1)
    # B[1, 2t] (V idx 6+2t), B[1, 2t+1] (V idx 7+2t)
    
    # C indices for block t:
    # C[0, 2t] (W idx 4t), C[0, 2t+1] (W idx 4t+2)
    # C[1, 2t] (W idx 4t+1), C[1, 2t+1] (W idx 4t+3)
    
    for t in range(3):
        # Strassen M1: (A00+A11)(B00+B11)
        U[0, col] = 1.0; U[4, col] = 1.0
        V[2*t, col] = 1.0; V[7+2*t, col] = 1.0
        W[4*t, col] = 1.0; W[4*t+3, col] = 1.0
        col += 1
        
        # Strassen M2: (A10+A11)B00
        U[3, col] = 1.0; U[4, col] = 1.0
        V[2*t, col] = 1.0
        W[4*t+1, col] = 1.0; W[4*t+3, col] = -1.0
        col += 1
        
        # Strassen M3: A00(B01-B11)
        U[0, col] = 1.0
        V[2*t+1, col] = 1.0; V[7+2*t, col] = -1.0
        W[4*t+2, col] = 1.0; W[4*t+3, col] = 1.0
        col += 1
        
        # Strassen M4: A11(B10-B00)
        U[4, col] = 1.0
        V[6+2*t, col] = 1.0; V[2*t, col] = -1.0
        W[4*t, col] = 1.0; W[4*t+1, col] = 1.0
        col += 1
        
        # Strassen M5: (A00+A01)B11
        U[0, col] = 1.0; U[1, col] = 1.0
        V[7+2*t, col] = 1.0
        W[4*t, col] = -1.0; W[4*t+2, col] = 1.0
        col += 1
        
        # Strassen M6: (A10-A00)(B00+B01)
        U[3, col] = 1.0; U[0, col] = -1.0
        V[2*t, col] = 1.0; V[2*t+1, col] = 1.0
        W[4*t+3, col] = 1.0
        col += 1
        
        # Strassen M7: (A01-A11)(B10+B11)
        U[1, col] = 1.0; U[4, col] = -1.0
        V[6+2*t, col] = 1.0; V[7+2*t, col] = 1.0
        W[4*t, col] = 1.0; W[4*t+3, col] = 1.0
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
