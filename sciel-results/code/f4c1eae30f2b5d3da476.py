import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Rank-44 decomposition: 4 blocks of M_{3,2,2} with rank 11
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local 3x2x2 decomposition (Rank 11)
    # A: 3x2, B: 2x2, C: 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part for rows 0,1 of A (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part for Row 2 of A (cols 7-10)
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def place_block(bi, ar0, ac0, br0, bc0, cr0, cc0):
        base = bi * 11
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = (ar0 + r) * m + (ac0 + c)
                U[gi, base:base+11] += U_loc[li, :]
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (br0 + r) * p + (bc0 + c)
                V[gi, base:base+11] += V_loc[li, :]
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = (cc0 + c) * n + (cr0 + r)
                W[gi, base:base+11] += W_loc[li, :]
    
    place_block(0, 0, 0, 0, 0, 0, 0)
    place_block(1, 0, 2, 2, 0, 0, 0)
    place_block(2, 0, 0, 0, 2, 0, 2)
    place_block(3, 0, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
