import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # W2 rows must be in order: C11, C21, C12, C22
    # C11 = M1 - M4 + M5 - M7
    # C21 = M2 + M4
    # C12 = M3 + M7
    # C22 = M1 - M2 + M3 + M6
    W2 = [
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [0, 0, 1, 0, 0, 0, 1],   # C12
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    U4 = np.kron(np.array(U2), np.array(U2))
    V4 = np.kron(np.array(V2), np.array(V2))
    W4 = np.kron(np.array(W2), np.array(W2))
    
    U4 = U4.tolist()
    V4 = V4.tolist()
    W4 = W4.tolist()
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # First block
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
            
    # Second block
    for i in range(16):
        for c in range(49):
            U[i][c + 49] = U4[i][c]
            V[i + 16][c + 49] = V4[i][c]
            W[i][c + 49] = W4[i][c]
            
    return {"U": U, "V": V, "W": W}
