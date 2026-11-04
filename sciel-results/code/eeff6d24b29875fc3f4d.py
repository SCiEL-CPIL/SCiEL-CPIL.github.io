import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build Strassen 4x4x4 (Rank 49)
    # Standard 2x2x2 Strassen
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    rank_444 = 49
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    # Block structure for 4x4x4
    # A is 4x4, split into 2x2 blocks of 2x2.
    # A_meta[c_meta] defines which 2x2 blocks of A are used in term c_meta.
    # Each block is a 2x2 matrix, expanded by U2.
    
    A_meta = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    B_meta = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    C_meta = [
        [(0, 0, 1), (1, 1, 1), (1, 0, -1), (0, 1, 1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, 1)],
    ]
    
    for c_meta in range(7):
        for c_sub in range(7):
            c = c_meta * 7 + c_sub
            for (bi, bj, coeff) in A_meta[c_meta]:
                for i in range(2):
                    for j in range(2):
                        # U index: (bi*2+i)*2 + (bj*2+j)
                        U444[(bi*2+i)*2 + (bj*2+j)][c] += coeff * U2[i*2+j][c_sub]
            for (bj, bk, coeff) in B_meta[c_meta]:
                for j in range(2):
                    for k in range(2):
                        # V index: (bj*2+j)*2 + (bk*2+k)
                        V444[(bj*2+j)*2 + (bk*2+k)][c] += coeff * V2[j*2+k][c_sub]
            for (ci, ck, coeff) in C_meta[c_meta]:
                for i in range(2):
                    for k in range(2):
                        # W index: (ck*2+k)*2 + (ci*2+i)
                        W444[(ck*2+k)*2 + (ci*2+i)][c] += coeff * W2[k*2+i][c_sub]
                        
    # 2. Extend to 4x4x5
    rank = 53
    U = [row[:] for row in U444]
    V = [row[:] for row in V444]
    W = [row[:] for row in W444]
    
    # Extend dimensions
    for i in range(16):
        U[i].extend([0]*4)
    for i in range(20):
        V[i].extend([0]*4)
    for i in range(20):
        W[i].extend([0]*4)
        
    # Add schoolbook terms for 5th column (k=4)
    for j in range(4):
        c = 49 + j
        # U[i, j, c] = 1 for all i
        for i in range(4):
            U[i*4 + j][c] = 1
        # V[j, 4, c] = 1
        V[j*5 + 4][c] = 1
        # W[4, i, c] = 1 for all i
        for i in range(4):
            W[4*4 + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
