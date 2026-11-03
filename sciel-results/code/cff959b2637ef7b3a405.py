import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Local vectors for 2x2x2 Strassen decomposition
    # Indices: 00, 01, 10, 11
    u_local = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_local = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_local = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [-1, 0, -1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, -1, 0]
    ]
    
    # 8 blocks of 2x2x2
    # i in {0,1}, j in {0,1,2,3}, k in {0..7}
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5), (6,7)
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for i_start in [0]: # Only i=0,1 form blocks. i=2 is leftover.
        for j_start, j_end in j_pairs:
            for k_start, k_end in k_pairs:
                for r in range(7):
                    # U
                    for li in range(4):
                        # Local index li corresponds to (li//2, li%2)
                        li_i = li // 2
                        li_j = li % 2
                        global_i = i_start + li_i
                        global_j = j_start + li_j
                        global_idx_u = global_i * m + global_j
                        U[global_idx_u, col] = u_local[r][li]
                    
                    # V
                    for lj in range(4):
                        lj_j = lj // 2
                        lj_k = lj % 2
                        global_j = j_start + lj_j
                        global_k = k_start + lj_k
                        global_idx_v = global_j * p + global_k
                        V[global_idx_v, col] = v_local[r][lj]
                    
                    # W
                    for lk in range(4):
                        lk_k = lk // 2
                        lk_i = lk % 2
                        global_k = k_start + lk_k
                        global_i = i_start + lk_i
                        global_idx_w = global_k * n + global_i
                        W[global_idx_w, col] = w_local[r][lk]
                    
                    col += 1
    
    # Remaining 32 terms: i=2, all j, all k
    for j in range(m):
        for k in range(p):
            global_i = 2
            global_idx_u = global_i * m + j
            global_idx_v = j * p + k
            global_idx_w = k * n + global_i
            
            U[global_idx_u, col] = 1
            V[global_idx_v, col] = 1
            W[global_idx_w, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
