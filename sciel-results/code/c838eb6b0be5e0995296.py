"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        return U, V, W
    
    U1, V1, W1 = strassen_222()
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            U_444[:, c] = np.kron(U1[:, c1], U1[:, c2])
            V_444[:, c] = np.kron(V1[:, c1], V1[:, c2])
            W_444[:, c] = np.kron(W1[:, c1], W1[:, c2])
    
    # 4x4x3 schoolbook
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48))
    W_443 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                U_443[i*4 + j, col] = 1
                V_443[j*3 + k_local, col] = 1
                W_443[k_local*4 + i, col] = 1
                col += 1
    
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    U[:, :49] = U_444
    V[:16, :49] = V_444
    W[:16, :49] = W_444
    
    U[:, 49:] = U_443
    # V for 4x4x3: j*3 + k_local → full index j*7 + (4+k_local)
    for col_idx in range(48):
        j = col_idx // 3
        k_local = col_idx % 3
        full_v_idx = j * 7 + (4 + k_local)
        V[full_v_idx, 49 + col_idx] = V_443[j*3 + k_local, col_idx]
    
    # W for 4x4x3: k_local*4 + i → full index (4+k_local)*4 + i
    for col_idx in range(48):
        k_local = col_idx // 4
        i = col_idx % 4
        full_w_idx = (4 + k_local) * 4 + i
        W[full_w_idx, 49 + col_idx] = W_443[k_local*4 + i, col_idx]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
