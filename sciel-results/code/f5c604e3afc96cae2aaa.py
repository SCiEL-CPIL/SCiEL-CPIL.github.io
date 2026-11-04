import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrices
    U_blk = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_blk = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_blk = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    U_sc = U_blk
    V_sc = V_blk
    W_sc = W_blk
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for k in range(7):
        for l in range(7):
            c = 7 * k + l
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    u = i % 2
                    J = j // 2
                    v = j % 2
                    r = 2 * I + J
                    u_prime = 2 * u + v
                    U[i * 4 + j, c] = U_blk[r, k] * U_sc[u_prime, l]
            
            for j in range(4):
                for k_idx in range(4):
                    J = j // 2
                    v = j % 2
                    K = k_idx // 2
                    w = k_idx % 2
                    s = 2 * J + K
                    v_prime = 2 * v + w
                    V[j * 4 + k_idx, c] = V_blk[s, k] * V_sc[v_prime, l]
            
            for k_idx in range(4):
                for i in range(4):
                    K = k_idx // 2
                    w = k_idx % 2
                    I = i // 2
                    u = i % 2
                    t = 2 * I + K
                    w_prime = 2 * u + w
                    W[k_idx * 4 + i, c] = W_blk[t, k] * W_sc[w_prime, l]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
