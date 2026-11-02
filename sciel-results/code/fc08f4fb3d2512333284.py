import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 1. Strassen terms for 2x2 block
    # Indices 0,1 for 2x2
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[4, 0] = -1; W[5, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    V[0, 1] = 1 # B00
    W[1, 1] = 1; W[5, 1] = 1 # C10, C11
    
    # Term 3
    U[0, 2] = 1 # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, B11
    W[4, 2] = 1; W[5, 2] = -1 # C01, C11
    
    # Term 4
    U[3, 3] = 1 # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, B00
    W[0, 3] = 1; W[4, 3] = -1 # C00, C01
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    V[3, 4] = 1 # B11
    W[0, 4] = -1; W[1, 4] = 1 # C00, C10
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1 # A10, A00
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    W[4, 5] = -1; W[1, 5] = 1 # C01, C10
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1 # A01, A11
    V[2, 6] = 1; V[3, 6] = 1 # B10, B11
    W[0, 6] = 1; W[5, 6] = -1 # C00, C11
    
    # 2. Standard terms for remaining 19 combinations
    # Indices involving 2 (0-based)
    # i,j,k in {0,1,2}
    # We need T_ijk = 1 for all i,j,k.
    # For i,j,k in {0,1}, the Strassen terms sum to 1.
    # For others, we add 1 term each.
    
    t = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * 3 + j, t] = 1
                V[j * 3 + k, t] = 1
                W[k * 3 + i, t] = 1
                t += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
