import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 15
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen U
    U_strassen = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T # Shape 4x7
    
    # Strassen V
    V_strassen = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T # Shape 4x7
    
    # Strassen W
    W_strassen = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ]).T # Shape 4x7
    
    # Block 1: k in {0, 1}
    U[:, 0:7] = U_strassen
    
    # V indices for block 1
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    V[0, 0:7] = V_strassen[0]
    V[1, 0:7] = V_strassen[1]
    V[5, 0:7] = V_strassen[2]
    V[6, 0:7] = V_strassen[3]
    
    # W indices for block 1
    # i=0, k=0 -> 0
    # i=0, k=1 -> 1
    # i=1, k=0 -> 2
    # i=1, k=1 -> 3
    W[0, 0:7] = W_strassen[0]
    W[1, 0:7] = W_strassen[1]
    W[2, 0:7] = W_strassen[2]
    W[3, 0:7] = W_strassen[3]
    
    # Block 2: k in {2, 3}
    U[:, 7:14] = U_strassen
    
    # V indices for block 2
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 7
    # j=1, k=3 -> 8
    V[2, 7:14] = V_strassen[0]
    V[3, 7:14] = V_strassen[1]
    V[7, 7:14] = V_strassen[2]
    V[8, 7:14] = V_strassen[3]
    
    # W indices for block 2
    # i=0, k=2 -> 4
    # i=0, k=3 -> 5
    # i=1, k=2 -> 6
    # i=1, k=3 -> 7
    W[4, 7:14] = W_strassen[0]
    W[5, 7:14] = W_strassen[1]
    W[6, 7:14] = W_strassen[2]
    W[7, 7:14] = W_strassen[3]
    
    # Block 3: k in {4}
    U[:, 14] = [1, 1, 1, 1]
    
    # V indices for block 3
    # j=0, k=4 -> 4
    # j=1, k=4 -> 9
    V[4, 14] = 1
    V[9, 14] = 1
    
    # W indices for block 3
    # i=0, k=4 -> 8
    # i=1, k=4 -> 9
    W[8, 14] = 1
    W[9, 14] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
