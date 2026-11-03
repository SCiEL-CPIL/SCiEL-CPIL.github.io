import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    U2_mat = U2.reshape(7, 2, 2)
    V2_mat = V2.reshape(7, 2, 2)
    W2_mat = W2.reshape(7, 2, 2)
    
    U4 = np.zeros((49, 16))
    V4 = np.zeros((49, 16))
    W4 = np.zeros((49, 16))
    
    for i in range(7):
        for j in range(7):
            idx = i * 7 + j
            U4[idx] = np.kron(U2_mat[i], U2_mat[j]).flatten()
            V4[idx] = np.kron(V2_mat[i], V2_mat[j]).flatten()
            W4[idx] = np.kron(W2_mat[i], W2_mat[j]).flatten()
            
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for k in range(49):
        U[:, k] = U4[k]
        U[:, 49+k] = U4[k]
        
        # V for block 1 (cols 0-3 of B)
        V[:16, k] = V4[k]
        # V for block 2 (cols 4-7 of B)
        V[16:32, 49+k] = V4[k]
        
        # W for block 1 (cols 0-3 of C)
        # W4 is flat 4x4 (i, k). W is (k, i).
        # W[k, i] = W4[i*4+k]
        M = W4[k].reshape(4, 4)
        W[:16, k] = M.T.flatten()
        
        # W for block 2 (cols 4-7 of C)
        W[16:32, 49+k] = M.T.flatten()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
