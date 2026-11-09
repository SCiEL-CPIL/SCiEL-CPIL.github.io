"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # 2x2x2 rank-7 Strassen decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0,  1, 0, 1, 0],
        [0, 0, 1,  0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    # Tensor square for 4x4x4: rank 49
    U4 = np.kron(U2, U2)  # 16 x 49
    V4 = np.kron(V2, V2)  # 16 x 49
    W4 = np.kron(W2, W2)  # 16 x 49
    
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # 4x4x4 part: columns 0..48
    U[:, 0:49] = U4
    
    # V for 4x4x4: V4 is 16x49, indexed by (j,k) for j,k in 0..3
    # In the full V (20x49), row index = j*5+k
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V4[j*4+k, 0:49]
    
    # W for 4x4x4: W4 is 16x49, indexed by (k,i) for k,i in 0..3
    # In the full W (20x49), row index = k*4+i
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # 4x4x1 part: columns 49..64
    # C[i] = sum_j A[i,j] * B[j,4]
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
            col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
