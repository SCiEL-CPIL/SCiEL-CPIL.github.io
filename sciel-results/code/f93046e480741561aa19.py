"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    # Rows correspond to (i,j) for U, (j,k) for V, (k,i) for W
    # i,j,k in {0,1}
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ])
    
    V2 = np.array([
        [1, 1, 0,-1, 0,  1, 0],
        [0, 0, 1, 0, 0,  1, 0],
        [0, 0, 0, 1, 0,  0, 1],
        [1, 0,-1, 0, 1,  0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1,-1,  0, 1],
        [0, 0, 1, 0, 1,  0, 0],
        [0, 1, 0, 1, 0,  0, 0],
        [1,-1, 1, 0, 0,  1, 0]
    ])
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Kronecker product construction
    for c1 in range(7):
        for c2 in range(7):
            c = 7 * c1 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    # U index: i*m + j
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    u_idx = i * m + j
                                    u_val = U2[2*i1 + j1, c1] * U2[2*i2 + j2, c2]
                                    if u_val != 0:
                                        U[u_idx, c] = u_val
                                    
                                    # V index: j*p + k
                                    k = 2 * k1 + k2
                                    v_idx = j * p + k
                                    v_val = V2[2*j1 + k1, c1] * V2[2*j2 + k2, c2]
                                    if v_val != 0:
                                        V[v_idx, c] = v_val
                                    
                                    # W index: k*n + i
                                    w_idx = k * n + i
                                    w_val = W2[2*k1 + i1, c1] * W2[2*k2 + i2, c2]
                                    if w_val != 0:
                                        W[w_idx, c] = w_val
                                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
