import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ])
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 0, 0, 0, 0, 1]
    ])
    
    # Schoolbook factors for 2x2x1
    SB_U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    SB_V = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    SB_W = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # U: i in 0..1, j in 0..1
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * m + (j + j_off)
                    U[u_idx, col] += S_U[i*2+j, r]
            # V: j in 0..1, k in 0..1
            for j in range(2):
                for k in range(2):
                    v_idx = (j + j_off) * p + (k + k_off)
                    V[v_idx, col] += S_V[j*2+k, r]
            # W: k in 0..1, i in 0..1
            for k in range(2):
                for i in range(2):
                    w_idx = (k + k_off) * n + (i + i_off)
                    W[w_idx, col] += S_W[k*2+i, r]
            col += 1

    def add_schoolbook(i_off, j_off, k_off):
        nonlocal col
        # U: i in 0..1, j in 0..1
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * m + (j + j_off)
                    U[u_idx, col] += SB_U[i*2+j, r]
            # V: j in 0..1, k is fixed at k_off
            for j in range(2):
                v_idx = (j + j_off) * p + k_off
                V[v_idx, col] += SB_V[j, r]
            # W: i in 0..1, k is fixed at k_off
            for i in range(2):
                w_idx = k_off * n + (i + i_off)
                W[w_idx, col] += SB_W[i, r]
            col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4]
    add_schoolbook(0, 0, 4)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4]
    add_schoolbook(0, 2, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
