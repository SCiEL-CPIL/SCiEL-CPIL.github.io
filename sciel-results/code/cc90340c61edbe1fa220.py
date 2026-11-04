import numpy as np

DIMENSIONS = [4, 4, 8]

# Strassen matrices for 2x2
U2 = np.array([
    [1, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1]
])

V2 = np.array([
    [1, 1, 0, -1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1]
])

W2 = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 1, 0]
])

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4
    rank_4x4 = 49
    rank_total = 2 * rank_4x4
    
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # Construct 4x4 decomposition
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # U4 rows: i, j
    for i in range(4):
        for j in range(4):
            r_u = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    U4[r_u, c] = U2[2*i1 + j1, u1] * U2[2*i2 + j2, u2]
                    
    # V4 rows: j, k
    for j in range(4):
        for k in range(4):
            r_v = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    V4[r_v, c] = V2[2*j1 + k1, u1] * V2[2*j2 + k2, u2]
                    
    # W4 rows: k, i
    for k in range(4):
        for i in range(4):
            r_w = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    W4[r_w, c] = W2[2*k1 + i1, u1] * W2[2*k2 + i2, u2]
                    
    # Assemble full U, V, W
    # U is same for both blocks
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    # V: Block 1 (cols 0-3 of B) -> V rows j*8 + k where k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
            
    # V: Block 2 (cols 4-7 of B) -> V rows j*8 + (k+4) where k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 8 + k + 4, 49:98] = V4[j * 4 + k, 0:49]
            
    # W: Block 1 (cols 0-3 of C) -> W rows k*4 + i where k in 0..3
    # W4 rows are k*4 + i. Indices match directly.
    W[0:16, 0:49] = W4
    
    # W: Block 2 (cols 4-7 of C) -> W rows (k+4)*4 + i where k in 0..3
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[k * 4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
