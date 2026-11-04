import numpy as np
import time

def construct():
    start_time = time.time()
    # Deadline guard
    deadline = start_time + 110.0 # 10s margin
    
    # The original build_block_strassen logic
    r_total = 1 + 3 * 7  # 22
    U = np.zeros((6, r_total), dtype=float)
    V = np.zeros((18, r_total), dtype=float)
    W = np.zeros((12, r_total), dtype=float)
    
    # Part 1: A1 * B1 (Rank 1)
    # A1 is A[:, 0] -> U[0], U[3]
    # B1 is B[0, :] -> V[0..5]
    # C1[i, k] = A[i, 0] * B[0, k]
    U[0, 0] = 1.0
    U[3, 0] = 1.0
    for k in range(6):
        V[k, 0] = 1.0
    for i in range(2):
        for k in range(6):
            W[k * 2 + i, 0] = 1.0
            
    # Part 2: A2 * B2 (Rank 21)
    # A2 is A[:, 1:3] -> U[1], U[2] (row 0), U[4], U[5] (row 1)
    # a11 = U[1], a12 = U[2], a21 = U[4], a22 = U[5]
    
    for t in range(3):
        base = 1 + t * 7
        # V indices for B2 block t
        # B2 is B[1:3, :]
        # Row 1 of B is V[6..11], Row 2 of B is V[12..17]
        v_b11 = 6 + 2 * t
        v_b12 = 7 + 2 * t
        v_b21 = 12 + 2 * t
        v_b22 = 13 + 2 * t
        
        # M1 = (a11 + a22) * (b11 + b22)
        U[1, base + 0] = 1.0
        U[5, base + 0] = 1.0
        V[v_b11, base + 0] = 1.0
        V[v_b22, base + 0] = 1.0
        # W: C0,2t (+1), C1,2t+1 (+1)
        W[4 * t, base + 0] = 1.0       # C[0, 2t] -> k=2t, i=0 -> idx 4t
        W[4 * t + 3, base + 0] = 1.0   # C[1, 2t+1] -> k=2t+1, i=1 -> idx (2t+1)*2+1 = 4t+3
        
        # M2 = (a21 + a22) * b11
        U[4, base + 1] = 1.0
        U[5, base + 1] = 1.0
        V[v_b11, base + 1] = 1.0
        # W: C1,2t (+1)
        W[4 * t + 1, base + 1] = 1.0   # C[1, 2t] -> k=2t, i=1 -> idx 4t+1
        
        # M3 = a11 * (b21 - b22)
        U[1, base + 2] = 1.0
        V[v_b21, base + 2] = 1.0
        V[v_b22, base + 2] = -1.0
        # W: C1,2t (+1), C1,2t+1 (-1)
        W[4 * t + 1, base + 2] = 1.0   # C[1, 2t]
        W[4 * t + 3, base + 2] = -1.0  # C[1, 2t+1]
        
        # M4 = a22 * (b12 - b11)
        U[5, base + 3] = 1.0
        V[v_b12, base + 3] = 1.0
        V[v_b11, base + 3] = -1.0
        # W: C0,2t+1 (+1)
        W[4 * t + 2, base + 3] = 1.0   # C[0, 2t+1] -> k=2t+1, i=0 -> idx 4t+2
        
        # M5 = (a11 + a12) * b22
        U[1, base + 4] = 1.0
        U[2, base + 4] = 1.0
        V[v_b22, base + 4] = 1.0
        # W: C0,2t (-1), C0,2t+1 (+1)
        W[4 * t, base + 4] = -1.0      # C[0, 2t]
        W[4 * t + 2, base + 4] = 1.0   # C[0, 2t+1]
        
        # M6 = (a21 - a11) * (b11 + b12)
        U[4, base + 5] = 1.0
        U[1, base + 5] = -1.0
        V[v_b11, base + 5] = 1.0
        V[v_b12, base + 5] = 1.0
        # W: C1,2t+1 (+1)
        W[4 * t + 3, base + 5] = 1.0   # C[1, 2t+1]
        
        # M7 = (a12 - a22) * (b21 + b22)
        U[2, base + 6] = 1.0
        U[5, base + 6] = -1.0
        V[v_b21, base + 6] = 1.0
        V[v_b22, base + 6] = 1.0
        # W: C0,2t (+1)
        W[4 * t, base + 6] = 1.0       # C[0, 2t]
        
    return {'U': U, 'V': V, 'W': W}
