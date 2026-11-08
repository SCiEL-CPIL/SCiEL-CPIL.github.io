import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 62
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # 1. A_11 (2x2) * B_11 (2x7) - Schoolbook (28 terms)
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # 2. A_12 (2x1) * B_21 (1x7) - Schoolbook (14 terms)
    # A_12 indices: i=0,1; j=2. U index = i*3+2.
    # B_21 indices: j=2; k=0..6. V index = 2*7+k = 14+k.
    # C_12 indices: i=0,1; k=0..6. W index = k*3+i.
    for i in range(2):
        for k in range(7):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # 3. A_21 (1x2) * B_11 (2x7) - Schoolbook (14 terms)
    # A_21 indices: i=2; j=0,1. U index = 2*3+j = 6+j.
    # B_11 indices: j=0,1; k=0..6. V index = j*7+k.
    # C_21 indices: i=2; k=0..6. W index = k*3+2.
    for j in range(2):
        for k in range(7):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
            
    # 4. A_22 (1x1) * B_21 (1x7) - Schoolbook (7 terms)
    # A_22 index: 8.
    # B_21 index: 14+k.
    # C_22 index: k*3+2.
    for k in range(7):
        U[8, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
